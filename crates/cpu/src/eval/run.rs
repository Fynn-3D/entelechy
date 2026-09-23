//! The machine that runs word code: one frame per call on a stack, the
//! memories' working copies, and what a trace records.

use orla_abi::Domain;
use orla_ir::{BinaryOp, Intrinsic};
use orla_rules::matrix::{self, Floats};

use crate::{ArithmeticError, ExecutionError};

use super::code::*;
use super::compile::*;

/// One call a traced run made: which function, and its frame as the
/// call left it.
pub(crate) struct CallRecord {
    pub(crate) function: usize,
    pub(crate) frame: Vec<u32>,
}

/// How many calls a trace records before it stops recording.
pub(crate) const CALL_RECORDS: usize = 256;

/// Interpreter state for one thread: a stack of words holding every
/// frame, and the loop indices of the running element.
pub(crate) struct Machine<'p, 's> {
    pub(super) program: &'p Program<'s>,
    /// The calls made while recording, for a trace; `None` when not.
    pub(super) calls: Option<Vec<CallRecord>>,
    /// Whether more calls happened than were recorded.
    pub(super) calls_truncated: bool,
    /// The memories' working copies, one per memory input: what the
    /// steps have written so far, and the outputs at the end.
    pub(super) memories: Vec<Option<Vec<u32>>>,
    pub(super) stack: Vec<u32>,
    /// Top of the stack: where the next call's frame begins.
    pub(super) sp: usize,
    pub(super) loops: Vec<i32>,
    pub(super) at: [u32; 3],
    pub(super) scratch: Vec<f32>,
}

pub(super) fn f(word: u32) -> f32 {
    f32::from_bits(word)
}

impl<'p, 's> Machine<'p, 's> {
    pub(crate) fn new(program: &'p Program<'s>) -> Self {
        let memories = program
            .inputs
            .iter()
            .zip(&program.memories)
            .map(|(view, memory)| memory.then(|| view.words.to_vec()))
            .collect();
        Self {
            program,
            calls: None,
            calls_truncated: false,
            memories,
            stack: vec![0; program.code.entry.frame * 2],
            sp: 0,
            loops: Vec::new(),
            at: [0; 3],
            scratch: Vec::new(),
        }
    }

    /// Runs the entry point for one element and writes its results into
    /// `out` as words, result after result.
    pub(crate) fn run_entry_words(
        &mut self,
        at: [u32; 3],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let entry = &program.code.entry;
        self.at = at;
        self.loops.clear();
        self.begin(entry.frame);
        self.run(&entry.code, 0)?;
        let mut cursor = 0;
        for result in &entry.results {
            out[cursor..cursor + result.n]
                .copy_from_slice(&self.stack[result.at..result.at + result.n]);
            cursor += result.n;
        }
        Ok(())
    }

    /// Runs the entry for element `index` of a scan with the carried value
    /// `carried`, writing the step's result (the next carried value) into
    /// `out`.
    pub(crate) fn run_scan_step(
        &mut self,
        index: u32,
        carried: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let entry = &program.code.entry;
        self.at = [index, 0, 0];
        self.loops.clear();
        self.loops.push(index as i32);
        self.begin(entry.frame);
        self.stack[..program.code.carry].copy_from_slice(carried);
        self.run(&entry.code, 0)?;
        let result = entry.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Runs the module's `init` and writes the carried value into `out`.
    pub(crate) fn run_init_words(&mut self, out: &mut [u32]) -> Result<(), ExecutionError> {
        let program = self.program;
        let init = program
            .code
            .init
            .as_ref()
            .expect("only scan modules have an init");
        self.at = [0; 3];
        self.loops.clear();
        self.begin(init.frame);
        self.run(&init.code, 0)?;
        let result = init.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Runs the module's `merge` on two encoded values and writes the
    /// result into `out`.
    pub(crate) fn run_merge_words(
        &mut self,
        a: &[u32],
        b: &[u32],
        out: &mut [u32],
    ) -> Result<(), ExecutionError> {
        let program = self.program;
        let merge = program
            .code
            .merge
            .as_ref()
            .expect("only reduce modules merge");
        self.at = [0; 3];
        self.loops.clear();
        self.begin(merge.frame);
        let (la, lb) = (merge.args[0], merge.args[1]);
        self.stack[la.at..la.at + la.n].copy_from_slice(a);
        self.stack[lb.at..lb.at + lb.n].copy_from_slice(b);
        self.run(&merge.code, 0)?;
        let result = merge.results[0];
        out.copy_from_slice(&self.stack[result.at..result.at + result.n]);
        Ok(())
    }

    /// Starts recording the calls made from here on, for a trace.
    pub(crate) fn record_calls(&mut self) {
        self.calls = Some(Vec::new());
        self.calls_truncated = false;
    }

    /// The calls recorded, and whether any went unrecorded.
    pub(crate) fn take_calls(&mut self) -> (Vec<CallRecord>, bool) {
        (self.calls.take().unwrap_or_default(), self.calls_truncated)
    }

    /// The words of value `id` of function `function` in a recorded frame.
    pub(crate) fn function_value(
        &self,
        function: usize,
        frame: &[u32],
        id: orla_ir::ValueId,
    ) -> Vec<u32> {
        let loc = self.program.code.functions[function].locs[id.0 as usize];
        frame[loc.at..loc.at + loc.n].to_vec()
    }

    /// The words of the `index`th result of function `function` in a
    /// recorded frame.
    pub(crate) fn function_result(&self, function: usize, frame: &[u32], index: usize) -> Vec<u32> {
        let loc = self.program.code.functions[function].results[index];
        frame[loc.at..loc.at + loc.n].to_vec()
    }

    /// Zeros the entry's frame, so a trace of one step shows nothing
    /// left over from the steps before it.
    pub(crate) fn clear_entry_frame(&mut self) {
        let frame = self.program.code.entry.frame;
        if self.stack.len() < frame {
            self.stack.resize(frame, 0);
        }
        self.stack[..frame].fill(0);
    }

    /// The words of the entry's value `id`, as the last run of the entry
    /// left them in its frame (the frame begins at the stack's start):
    /// the value's last assignment, or zeros if its block never ran.
    pub(crate) fn entry_value(&self, id: orla_ir::ValueId) -> Vec<u32> {
        let loc = self.program.code.entry.locs[id.0 as usize];
        self.stack[loc.at..loc.at + loc.n].to_vec()
    }

    /// The words of the entry's `index`th result after a run.
    pub(crate) fn entry_result(&self, index: usize) -> Vec<u32> {
        let loc = self.program.code.entry.results[index];
        self.stack[loc.at..loc.at + loc.n].to_vec()
    }

    /// The memories as the steps left them, by input index.
    pub(crate) fn take_memories(&mut self) -> Vec<(usize, Vec<u32>)> {
        self.memories
            .iter_mut()
            .enumerate()
            .filter_map(|(index, memory)| memory.take().map(|words| (index, words)))
            .collect()
    }

    /// Opens the outermost frame of `words`.
    fn begin(&mut self, words: usize) {
        if self.stack.len() < words {
            self.stack.resize(words, 0);
        }
        self.sp = words;
    }

    fn arithmetic(&self, error: ArithmeticError) -> ExecutionError {
        let [x, y, z] = self.at;
        ExecutionError::Arithmetic {
            x,
            y: z * self.program.domain.height() + y,
            error,
        }
    }

    /// Runs `code` with its frame at `fp`.
    fn run(&mut self, code: &[Instr], fp: usize) -> Result<(), ExecutionError> {
        for instr in code {
            self.step(instr, fp)?;
        }
        Ok(())
    }

    fn step(&mut self, instr: &Instr, fp: usize) -> Result<(), ExecutionError> {
        let program = self.program;
        let s = &mut self.stack;
        match instr {
            Instr::Const { dst, words, n } => {
                s[fp + dst..fp + dst + n].copy_from_slice(&words[..*n]);
            }
            Instr::Words { dst, words } => {
                s[fp + dst..fp + dst + words.len()].copy_from_slice(words);
            }
            Instr::Param { dst, at, n } => {
                s[fp + dst..fp + dst + n].copy_from_slice(&program.params[*at..at + n]);
            }
            Instr::Move { dst, src, n } => {
                if src != dst {
                    s.copy_within(fp + src..fp + src + n, fp + dst);
                }
            }
            Instr::Gather { dst, parts } => {
                let mut cursor = fp + dst;
                for part in parts {
                    s.copy_within(fp + part.at..fp + part.at + part.n, cursor);
                    cursor += part.n;
                }
            }
            Instr::Subscript {
                dst,
                base,
                index,
                count,
                n,
            } => {
                let k = (s[fp + index] as i32).clamp(0, *count as i32 - 1) as usize;
                s.copy_within(fp + base + k * n..fp + base + (k + 1) * n, fp + dst);
            }
            Instr::MatMul { kind, dst, a, b, n } => {
                let floats = |at: usize, count: usize| -> Vec<f32> {
                    s[fp + at..fp + at + count].iter().map(|w| f(*w)).collect()
                };
                let square = n * n;
                let out = match kind {
                    MatMul::MatMat => matrix::mul_mat_mat(
                        &mut Floats,
                        *n,
                        &floats(*a, square),
                        &floats(*b, square),
                    ),
                    MatMul::MatVec => {
                        matrix::mul_mat_vec(&mut Floats, *n, &floats(*a, square), &floats(*b, *n))
                    }
                    MatMul::VecMat => {
                        matrix::mul_vec_mat(&mut Floats, *n, &floats(*a, *n), &floats(*b, square))
                    }
                };
                for (k, v) in out.iter().enumerate() {
                    s[fp + dst + k] = v.to_bits();
                }
            }
            Instr::Matrix { op, dst, src, n } => {
                let m: Vec<f32> = s[fp + src..fp + src + n * n]
                    .iter()
                    .map(|w| f(*w))
                    .collect();
                let out = match op {
                    Intrinsic::Transpose => matrix::transpose(*n, &m),
                    Intrinsic::Inverse => matrix::inverse(&mut Floats, *n, &m),
                    _ => vec![matrix::determinant(&mut Floats, *n, &m)],
                };
                for (k, v) in out.iter().enumerate() {
                    s[fp + dst + k] = v.to_bits();
                }
            }
            Instr::Vector { dst, parts } => {
                for (k, (src, widen)) in parts.iter().enumerate() {
                    let word = s[fp + src];
                    s[fp + dst + k] = match widen {
                        Widen::Float => word,
                        Widen::Int => (word as i32 as f32).to_bits(),
                    };
                }
            }
            Instr::FNeg { dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = (-f(s[fp + src + i])).to_bits();
                }
            }
            Instr::INeg { dst, src, n } => {
                // The frame is read through `self` here: a failure is
                // reported through `self` too, so the borrow of the stack
                // cannot live across the iterations.
                for i in 0..*n {
                    let v = (self.stack[fp + src + i] as i32)
                        .checked_neg()
                        .ok_or(ArithmeticError::Overflow)
                        .map_err(|e| self.arithmetic(e))?;
                    self.stack[fp + dst + i] = v as u32;
                }
            }
            Instr::Not { dst, src } => s[fp + dst] = (s[fp + src] == 0) as u32,
            Instr::FBin {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + a + i * sa]), f(s[fp + b + i * sb])).to_bits();
                }
            }
            Instr::FBin2 {
                f: inner,
                g: outer,
                flip,
                dst,
                a,
                sa,
                b,
                sb,
                c,
                sc,
                n,
            } => {
                if *flip {
                    for i in 0..*n {
                        let x = inner(f(s[fp + a + i * sa]), f(s[fp + b + i * sb]));
                        s[fp + dst + i] = outer(f(s[fp + c + i * sc]), x).to_bits();
                    }
                } else {
                    for i in 0..*n {
                        let x = inner(f(s[fp + a + i * sa]), f(s[fp + b + i * sb]));
                        s[fp + dst + i] = outer(x, f(s[fp + c + i * sc])).to_bits();
                    }
                }
            }
            Instr::IBin {
                op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    let (a, b) = (
                        self.stack[fp + a + i * sa] as i32,
                        self.stack[fp + b + i * sb] as i32,
                    );
                    let result = match op {
                        BinaryOp::Add => a.checked_add(b).ok_or(ArithmeticError::Overflow),
                        BinaryOp::Sub => a.checked_sub(b).ok_or(ArithmeticError::Overflow),
                        BinaryOp::Mul => a.checked_mul(b).ok_or(ArithmeticError::Overflow),
                        BinaryOp::Div if b == 0 => Err(ArithmeticError::DivisionByZero),
                        BinaryOp::Div => a.checked_div(b).ok_or(ArithmeticError::Overflow),
                        BinaryOp::Rem if b == 0 => Err(ArithmeticError::DivisionByZero),
                        BinaryOp::Rem => a.checked_rem(b).ok_or(ArithmeticError::Overflow),
                        BinaryOp::BitAnd => Ok(a & b),
                        BinaryOp::BitOr => Ok(a | b),
                        BinaryOp::BitXor => Ok(a ^ b),
                        // The count modulo 32, as the GPU masks it.
                        BinaryOp::Shl => Ok(a.wrapping_shl(b as u32)),
                        BinaryOp::Shr => Ok(a.wrapping_shr(b as u32)),
                        BinaryOp::Ushr => Ok((a as u32).wrapping_shr(b as u32) as i32),
                        _ => unreachable!("compiled from an arithmetic operator"),
                    };
                    let v = result.map_err(|e| self.arithmetic(e))?;
                    self.stack[fp + dst + i] = v as u32;
                }
            }
            Instr::FCmp { op, dst, a, b } => {
                let (a, b) = (f(s[fp + a]), f(s[fp + b]));
                // NaN compares false, as in IEEE 754 and every GPU language.
                let holds = match op {
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    _ => a >= b,
                };
                s[fp + dst] = holds as u32;
            }
            Instr::ICmp { op, dst, a, b } => {
                let (a, b) = (s[fp + a] as i32, s[fp + b] as i32);
                let holds = match op {
                    BinaryOp::Less => a < b,
                    BinaryOp::LessEqual => a <= b,
                    BinaryOp::Greater => a > b,
                    _ => a >= b,
                };
                s[fp + dst] = holds as u32;
            }
            Instr::FEq {
                dst,
                a,
                b,
                n,
                negate,
            } => {
                let equal = (0..*n).all(|i| f(s[fp + a + i]) == f(s[fp + b + i]));
                s[fp + dst] = (equal != *negate) as u32;
            }
            Instr::IEq {
                dst,
                a,
                b,
                n,
                negate,
            } => {
                let same = (0..*n).all(|i| s[fp + a + i] == s[fp + b + i]);
                s[fp + dst] = (same != *negate) as u32;
            }
            Instr::And { dst, a, b } => s[fp + dst] = (s[fp + a] != 0 && s[fp + b] != 0) as u32,
            Instr::Or { dst, a, b } => s[fp + dst] = (s[fp + a] != 0 || s[fp + b] != 0) as u32,
            Instr::Map1 { f: op, dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + src + i])).to_bits();
                }
            }
            Instr::Map2 {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(f(s[fp + a + i * sa]), f(s[fp + b + i * sb])).to_bits();
                }
            }
            Instr::Map3 {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                c,
                sc,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(
                        f(s[fp + a + i * sa]),
                        f(s[fp + b + i * sb]),
                        f(s[fp + c + i * sc]),
                    )
                    .to_bits();
                }
            }
            Instr::IMap1 { f: op, dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = op(s[fp + src + i] as i32) as u32;
                }
            }
            Instr::IMap2Checked { f: op, dst, a, b } => {
                let v = op(s[fp + a] as i32, s[fp + b] as i32).map_err(|e| self.arithmetic(e))?;
                self.stack[fp + dst] = v as u32;
            }
            Instr::IMap2 {
                f: op,
                dst,
                a,
                sa,
                b,
                sb,
                n,
            } => {
                for i in 0..*n {
                    s[fp + dst + i] =
                        op(s[fp + a + i * sa] as i32, s[fp + b + i * sb] as i32) as u32;
                }
            }
            Instr::IClamp {
                dst,
                v,
                low,
                sl,
                high,
                sh,
                n,
            } => {
                for i in 0..*n {
                    let (v, low, high) = (
                        s[fp + v + i] as i32,
                        s[fp + low + i * sl] as i32,
                        s[fp + high + i * sh] as i32,
                    );
                    s[fp + dst + i] = v.max(low).min(high) as u32;
                }
            }
            Instr::ToFloat { dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = (s[fp + src + i] as i32 as f32).to_bits();
                }
            }
            Instr::ToInt { dst, src, n } => {
                for i in 0..*n {
                    s[fp + dst + i] = (f(s[fp + src + i]) as i32) as u32;
                }
            }
            Instr::Dot { dst, a, b, n } => {
                let v = (0..*n).fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + b + i]));
                s[fp + dst] = v.to_bits();
            }
            Instr::Cross { dst, a, b } => {
                let a = [f(s[fp + a]), f(s[fp + a + 1]), f(s[fp + a + 2])];
                let b = [f(s[fp + b]), f(s[fp + b + 1]), f(s[fp + b + 2])];
                s[fp + dst] = (a[1] * b[2] - a[2] * b[1]).to_bits();
                s[fp + dst + 1] = (a[2] * b[0] - a[0] * b[2]).to_bits();
                s[fp + dst + 2] = (a[0] * b[1] - a[1] * b[0]).to_bits();
            }
            Instr::Length { dst, a, n } => {
                let v = (0..*n).fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + a + i]));
                s[fp + dst] = v.sqrt().to_bits();
            }
            Instr::Normalize { dst, a, n } => {
                let length = (0..*n)
                    .fold(0.0f32, |sum, i| sum + f(s[fp + a + i]) * f(s[fp + a + i]))
                    .sqrt();
                for i in 0..*n {
                    let v = f(s[fp + a + i]);
                    s[fp + dst + i] = if length > 0.0 { v / length } else { 0.0 }.to_bits();
                }
            }
            Instr::Reflect {
                dst,
                incident,
                normal,
                n,
            } => {
                let scale = 2.0
                    * (0..*n).fold(0.0f32, |sum, i| {
                        sum + f(s[fp + normal + i]) * f(s[fp + incident + i])
                    });
                for i in 0..*n {
                    s[fp + dst + i] =
                        (f(s[fp + incident + i]) - scale * f(s[fp + normal + i])).to_bits();
                }
            }
            Instr::Atan2 { dst, y, x } => s[fp + dst] = f(s[fp + y]).atan2(f(s[fp + x])).to_bits(),
            Instr::If {
                condition,
                then,
                then_result,
                otherwise,
                otherwise_result,
                dst,
                n,
            } => {
                let result = if s[fp + condition] != 0 {
                    self.run(then, fp)?;
                    *then_result
                } else {
                    self.run(otherwise, fp)?;
                    *otherwise_result
                };
                if result != *dst {
                    self.stack
                        .copy_within(fp + result..fp + result + n, fp + dst);
                }
            }
            Instr::Match {
                scrutinee,
                arms,
                otherwise,
                otherwise_result,
                dst,
                n,
            } => {
                let value = s[fp + scrutinee] as i32;
                let (code, result) = arms
                    .iter()
                    .find(|(values, _, _)| values.contains(&value))
                    .map_or((otherwise, *otherwise_result), |(_, code, result)| {
                        (code, *result)
                    });
                self.run(code, fp)?;
                if result != *dst {
                    self.stack
                        .copy_within(fp + result..fp + result + n, fp + dst);
                }
            }
            Instr::Loop {
                count,
                bound,
                init,
                carried,
                n,
                until,
                body,
                body_result,
            } => {
                s.copy_within(fp + init..fp + init + n, fp + carried);
                let times = match bound {
                    Some(bound) => (s[fp + bound] as i32).clamp(0, *count as i32) as u32,
                    None => *count,
                };
                self.loops.push(0);
                for index in 0..times {
                    *self.loops.last_mut().expect("loop is open") = index as i32;
                    if let Some((until, done)) = until {
                        self.run(until, fp)?;
                        if self.stack[fp + done] != 0 {
                            break;
                        }
                    }
                    self.run(body, fp)?;
                    self.stack
                        .copy_within(fp + body_result..fp + body_result + n, fp + carried);
                }
                self.loops.pop();
            }
            Instr::Index { dst } => {
                s[fp + dst] = *self.loops.last().expect("validation guarantees a loop") as u32;
            }
            Instr::Call {
                function,
                args,
                dst,
                n,
            } => {
                let callee = &program.code.functions[*function];
                let base = self.sp;
                if s.len() < base + callee.frame {
                    s.resize(base + callee.frame, 0);
                }
                for (arg, param) in args.iter().zip(&callee.args) {
                    s.copy_within(fp + arg.at..fp + arg.at + arg.n, base + param.at);
                }
                self.sp = base + callee.frame;
                self.run(&callee.code, base)?;
                if let Some(records) = &mut self.calls {
                    if records.len() < CALL_RECORDS {
                        records.push(CallRecord {
                            function: *function,
                            frame: self.stack[base..base + callee.frame].to_vec(),
                        });
                    } else {
                        self.calls_truncated = true;
                    }
                }
                let result = callee.results[0];
                self.stack
                    .copy_within(base + result.at..base + result.at + n, fp + dst);
                self.sp = base;
            }
            Instr::Coord { dst, n } => {
                for (i, axis) in self.at.iter().enumerate().take(*n) {
                    s[fp + dst + i] = (*axis as f32).to_bits();
                }
            }
            Instr::Extent { dst } => write_extent(s, fp + dst, program.domain),
            Instr::Element { dst } => {
                let [x, y, z] = self.at;
                let domain = program.domain;
                s[fp + dst] = (z * domain.height() + y) * domain.width() + x;
            }
            Instr::Time { dst } => s[fp + dst] = program.context.time.to_bits(),
            Instr::Rate { dst } => s[fp + dst] = program.context.rate.to_bits(),
            Instr::Pointer { dst } => {
                let [x, y] = program.context.input.pointer;
                s[fp + dst] = x.to_bits();
                s[fp + dst + 1] = y.to_bits();
            }
            Instr::PointerDown { dst } => s[fp + dst] = program.context.input.pointer_down as u32,
            Instr::Key { dst, code } => {
                s[fp + dst] = program.context.input.key(s[fp + code] as i32) as u32;
            }
            Instr::Button { dst, code } => {
                s[fp + dst] = program.context.input.button(s[fp + code] as i32) as u32;
            }
            Instr::Stick { dst, which } => {
                let [x, y] = program.context.input.stick(s[fp + which] as i32);
                s[fp + dst] = x.to_bits();
                s[fp + dst + 1] = y.to_bits();
            }
            Instr::Wheel { dst } => {
                let [x, y] = program.context.input.wheel;
                s[fp + dst] = x.to_bits();
                s[fp + dst + 1] = y.to_bits();
            }
            Instr::Window { dst } => {
                let [w, h] = program.context.window;
                s[fp + dst] = w.to_bits();
                s[fp + dst + 1] = h.to_bits();
            }
            Instr::Scale { dst } => s[fp + dst] = program.context.scale.to_bits(),
            Instr::Count { dst, input } => {
                s[fp + dst] = program.inputs[*input].domain.count() as u32;
            }
            Instr::Size { dst, input } => write_extent(s, fp + dst, program.inputs[*input].domain),
            Instr::Sample { dst, input, n } => {
                let view = &program.inputs[*input];
                let index = own_index(self.at, view.domain);
                let words = self.memories[*input].as_deref().unwrap_or(view.words);
                view.read(words, index, 0, &mut s[fp + dst..fp + dst + n]);
            }
            Instr::At {
                dst,
                input,
                address,
                kind,
                n,
            } => {
                let view = &program.inputs[*input];
                let index = address_index(s, fp + address, *kind, view.domain);
                let words = self.memories[*input].as_deref().unwrap_or(view.words);
                view.read(words, index, 0, &mut s[fp + dst..fp + dst + n]);
            }
            Instr::Part {
                dst,
                input,
                address,
                base,
                steps,
                n,
            } => {
                let view = &program.inputs[*input];
                let index = match address {
                    Some((at, kind)) => address_index(s, fp + at, *kind, view.domain),
                    None => own_index(self.at, view.domain),
                };
                let words = self.memories[*input].as_deref().unwrap_or(view.words);
                // Where the path leads: the words it is known to pass
                // over, then each index clamped into what it indexes.
                let mut offset = *base;
                for (slot, count, stride) in steps {
                    let k = (s[fp + slot] as i32).clamp(0, *count as i32 - 1) as usize;
                    offset += k * stride;
                }
                let first = offset.min(view.stride.saturating_sub(*n));
                view.read(words, index, first, &mut s[fp + dst..fp + dst + n]);
            }
            Instr::Put {
                dst,
                input,
                address,
                kind,
                value,
                n,
            } => {
                // The value written is the instruction's own value; an
                // address outside the memory is dropped, never clamped.
                let view = &program.inputs[*input];
                let written = fp + value;
                s.copy_within(written..written + n, fp + dst);
                let index = match kind {
                    Address::Int => {
                        let i = s[fp + address] as i32;
                        (i >= 0 && (i as usize) < view.domain.count()).then_some(i as usize)
                    }
                    _ => {
                        let address = read_address(s, fp + address, *kind, 0.0);
                        let words: Vec<u32> = address.iter().map(|v| v.to_bits()).collect();
                        crate::scatter_index(view.domain, &words)
                    }
                };
                if let (Some(index), Some(memory)) = (index, self.memories[*input].as_mut()) {
                    view.write(memory, index, &s[written..written + n]);
                }
            }
            Instr::AtLinear {
                dst,
                input,
                position,
                kind,
                n,
            } => {
                let view = &program.inputs[*input];
                let address = read_address(s, fp + position, *kind, 0.5);
                let domain = view.domain;
                // Element centres sit at half-integers: the position is
                // moved by a half, split into the lower neighbour and a
                // weight, and each axis is clamped at the edges.
                let axis = |v: f32, limit: u32| -> (usize, usize, f32) {
                    let t = if v.is_nan() { 0.0 } else { v - 0.5 };
                    let lower = t.floor().clamp(0.0, (limit - 1) as f32);
                    let upper = (lower + 1.0).min((limit - 1) as f32);
                    let weight = (t - t.floor()).clamp(0.0, 1.0);
                    let weight = if t < 0.0 { 0.0 } else { weight };
                    (lower as usize, upper as usize, weight)
                };
                let (x0, x1, fx) = axis(address[0], domain.width());
                let (y0, y1, fy) = axis(address[1], domain.height());
                let (z0, z1, fz) = axis(address[2], domain.depth());
                self.scratch.clear();
                self.scratch.resize(*n, 0.0);
                let (width, height) = (domain.width() as usize, domain.height() as usize);
                for (corner, weight) in [
                    ((x0, y0, z0), (1.0 - fx) * (1.0 - fy) * (1.0 - fz)),
                    ((x1, y0, z0), fx * (1.0 - fy) * (1.0 - fz)),
                    ((x0, y1, z0), (1.0 - fx) * fy * (1.0 - fz)),
                    ((x1, y1, z0), fx * fy * (1.0 - fz)),
                    ((x0, y0, z1), (1.0 - fx) * (1.0 - fy) * fz),
                    ((x1, y0, z1), fx * (1.0 - fy) * fz),
                    ((x0, y1, z1), (1.0 - fx) * fy * fz),
                    ((x1, y1, z1), fx * fy * fz),
                ] {
                    if weight == 0.0 {
                        continue;
                    }
                    let (cx, cy, cz) = corner;
                    let index = (cz * height + cy) * width + cx;
                    let words = self.memories[*input].as_deref().unwrap_or(view.words);
                    for (k, slot) in self.scratch.iter_mut().enumerate() {
                        let word = match &view.narrow {
                            None => words[index * view.stride + k],
                            Some(schema) => schema.load(words, index, k),
                        };
                        *slot += f(word) * weight;
                    }
                }
                for (i, v) in self.scratch.iter().enumerate() {
                    self.stack[fp + dst + i] = v.to_bits();
                }
            }
        }
        Ok(())
    }
}

/// Writes a domain's extent as the value its shape is addressed with.
pub(super) fn write_extent(stack: &mut [u32], at: usize, domain: Domain) {
    match domain {
        Domain::Rows(n) => stack[at] = n,
        Domain::Grid(w, h) => {
            stack[at] = (w as f32).to_bits();
            stack[at + 1] = (h as f32).to_bits();
        }
        Domain::Volume(w, h, d) => {
            stack[at] = (w as f32).to_bits();
            stack[at + 1] = (h as f32).to_bits();
            stack[at + 2] = (d as f32).to_bits();
        }
    }
}

/// Reads an address of the given kind; missing axes take `rest`.
/// This element's own place in an input, clamped into it: what a
/// `sample` reads, and a part of one.
#[inline]
pub(super) fn own_index([x, y, z]: [u32; 3], domain: Domain) -> usize {
    linear(
        domain,
        [
            x.min(domain.width() - 1),
            y.min(domain.height() - 1),
            z.min(domain.depth() - 1),
        ],
    )
}

/// The element an address names, clamped into an input: a row's
/// address is one `Int` and one clamp, which is what most reads are;
/// the rest go by axis through floats.
#[inline]
pub(super) fn address_index(stack: &[u32], at: usize, kind: Address, domain: Domain) -> usize {
    match kind {
        Address::Int => {
            let last = domain.count() as i32 - 1;
            (stack[at] as i32).clamp(0, last) as usize
        }
        _ => linear(
            domain,
            clamp_address(domain, read_address(stack, at, kind, 0.0)),
        ),
    }
}

pub(super) fn read_address(stack: &[u32], at: usize, kind: Address, rest: f32) -> [f32; 3] {
    match kind {
        Address::Int => [stack[at] as i32 as f32, rest, rest],
        Address::Float => [f(stack[at]), rest, rest],
        Address::Vec2 => [f(stack[at]), f(stack[at + 1]), rest],
        Address::Vec3 => [f(stack[at]), f(stack[at + 1]), f(stack[at + 2])],
    }
}

/// Each axis floored and clamped into range (NaN counts as zero), so no
/// read is ever outside.
pub(super) fn clamp_address(domain: Domain, address: [f32; 3]) -> [u32; 3] {
    let clamp = |value: f32, limit: u32| -> u32 {
        if value.is_nan() {
            0
        } else {
            value.floor().clamp(0.0, (limit - 1) as f32) as u32
        }
    };
    [
        clamp(address[0], domain.width()),
        clamp(address[1], domain.height()),
        clamp(address[2], domain.depth()),
    ]
}

/// The linear index of an in-range address.
pub(super) fn linear(domain: Domain, [x, y, z]: [u32; 3]) -> usize {
    ((z * domain.height() + y) * domain.width() + x) as usize
}
