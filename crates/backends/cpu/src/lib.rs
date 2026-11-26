#![warn(missing_docs)]
//! CPU backend.
//!
//! Compiles a [`Validated`] module once into a [`Code`] the caller keeps,
//! and runs it once per output element. Every Host access goes through
//! the capability-gated [`Scope`]; a failed execution releases the
//! outputs it created.
//!
//! The compute phase borrows inputs through the scope once and runs rows
//! in parallel; the write phase stores results sequentially. Results equal
//! a single-threaded run, and the first failing element in row-major
//! order is the one reported.

mod eval;

use std::any::Any;

use entelechy_abi::{Domain, Handle, Invocation, Scope, Type};
use entelechy_ir::{Combine, Mode, Validated};

use entelechy_rules::{Backend, NativeOp, Prepared};

pub use entelechy_rules::{
    ArithmeticError, ExecutionError, address_words, check_invocation, check_scan, check_scatter,
    combine_words, float_order_key, scatter_index,
};

/// The CPU backend. Stateless; one instance can execute any module.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

/// A module compiled to the interpreter's word code, compiled once and
/// bound to each invocation.
#[derive(Clone)]
pub struct Code(std::sync::Arc<eval::Compiled>);

impl Code {
    /// Compiles a module; a group's entry is first rewritten to run its
    /// lanes ([`entelechy_ir::lanes`]).
    pub fn of(module: &Validated) -> Result<Self, String> {
        let lowered = entelechy_ir::lanes(module).map_err(|error| error.to_string())?;
        Ok(Self(std::sync::Arc::new(eval::Program::compile(&lowered))))
    }
}

impl CpuBackend {
    /// Executes `module` over `invocation.domain`, returning one handle per
    /// manifest output. The handles are owned by the caller's Host.
    pub fn execute(
        &self,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.execute_with(None, module, invocation, scope)
    }

    /// The same, with the module's word code already compiled.
    pub fn execute_with(
        &self,
        code: Option<&Code>,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        check_invocation(module, invocation, scope)?;
        let code = match code {
            Some(code) => code.0.clone(),
            None => Code::of(module).map_err(ExecutionError::Device)?.0,
        };
        let strides = output_strides(module);
        if module.manifest.mode == Mode::Reduce {
            return reduce(&code, module, invocation, scope, strides[0]);
        }
        if module.manifest.mode == Mode::Scan {
            return scan(&code, module, invocation, scope, strides[0]);
        }
        if let Mode::Prefix { combine } = module.manifest.mode {
            return prefix(&code, module, invocation, scope, strides[0], combine);
        }
        if let Mode::Scatter { base, combine } = module.manifest.mode {
            return scatter(
                &code,
                module,
                invocation,
                scope,
                strides[0],
                base as usize,
                combine,
            );
        }
        entelechy_rules::outputs_fit_words(module, invocation.domain, scope)?;
        let program = eval::Program::with(code, module, invocation, scope, invocation.domain)?;
        let results = match module.manifest.mode {
            Mode::Group { size } => compute_group(&program, &strides, size as usize)?,
            _ => compute(&program, &strides)?,
        };
        program.outcome()?;
        drop(program);
        let domain = invocation.domain;
        let stride: usize = strides.iter().sum();
        let count = domain.count();
        // With one output, the elements' words are already the Field's
        // words in order and go in whole.
        if let [port] = &module.manifest.outputs[..] {
            let handle = scope.create(port.ty.schema.clone(), domain)?;
            return match scope.write_values(handle, &results) {
                Ok(()) => Ok(vec![handle]),
                Err(error) => {
                    let _ = scope.release(handle);
                    Err(error.into())
                }
            };
        }
        let mut outputs = Vec::with_capacity(module.manifest.outputs.len());
        let mut offset = 0;
        for (port, words) in module.manifest.outputs.iter().zip(&strides) {
            // Gather this output's words, element after element.
            let mut own: Vec<u32> = Vec::with_capacity(count * words);
            for element in 0..count {
                let base = element * stride + offset;
                own.extend_from_slice(&results[base..base + words]);
            }
            offset += words;
            let created = scope
                .create(port.ty.schema.clone(), domain)
                .and_then(|handle| scope.write_values(handle, &own).map(|()| handle));
            match created {
                Ok(handle) => outputs.push(handle),
                Err(error) => {
                    release_all(scope, &outputs);
                    return Err(error.into());
                }
            }
        }
        Ok(outputs)
    }
}

/// Words per element of each output: its schema's count.
fn output_strides(module: &Validated) -> Vec<usize> {
    module
        .manifest
        .outputs
        .iter()
        .map(|port| port.ty.schema.value_words())
        .collect()
}

/// A reduction: the entry over the first input's domain, then a pairwise
/// fold with `merge` in a fixed tree, independent of thread count.
/// Produces one `Rows` element.
fn reduce(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    let held = entelechy_rules::Held::new(scope.room());
    if !held.take(entelechy_rules::held_values(true, domain.count(), stride)) {
        return Err(held.over());
    }
    let program = eval::Program::with(code.clone(), module, invocation, scope, domain)?;
    let mut level = compute(&program, &[stride])?;
    let mut machine = eval::Machine::new(&program);
    let mut count = domain.count();
    while count > 1 {
        let pairs = count / 2;
        let mut next = vec![0u32; count.div_ceil(2) * stride];
        for pair in 0..pairs {
            let (a, b) = (
                &level[2 * pair * stride..(2 * pair + 1) * stride],
                &level[(2 * pair + 1) * stride..(2 * pair + 2) * stride],
            );
            machine.run_merge_words(a, b, &mut next[pair * stride..(pair + 1) * stride])?;
        }
        if count % 2 == 1 {
            next[pairs * stride..].copy_from_slice(&level[(count - 1) * stride..count * stride]);
        }
        level = next;
        count = count.div_ceil(2);
    }
    machine.finish()?;
    program.outcome()?;
    drop(machine);
    drop(program);
    let port = &module.manifest.outputs[0];
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_values(output, &level) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// A prefix: the entry over every element of the first input, then each
/// element folded with everything before it, in element order. The rule
/// is order-free, so every backend's answer equals this walk's.
fn prefix(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    let held = entelechy_rules::Held::new(scope.room());
    if !held.take(entelechy_rules::held_values(false, domain.count(), stride)) {
        return Err(held.over());
    }
    let program = eval::Program::with(code.clone(), module, invocation, scope, domain)?;
    let mut values = compute(&program, &[stride])?;
    program.outcome()?;
    drop(program);
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars();
    let count = domain.count();
    if count > 1 {
        let mut running = values[..stride].to_vec();
        for element in 1..count {
            let at = element * stride;
            entelechy_rules::combine_words(
                combine,
                scalars,
                &mut running,
                &values[at..at + stride],
            )
            .map_err(|error| ExecutionError::Arithmetic {
                x: element as u32,
                y: 0,
                error,
            })?;
            values[at..at + stride].copy_from_slice(&running);
        }
    }
    // The output has the input's count.
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_values(output, &values) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// One named value of a traced element: what a `let` held when the
/// element finished (its last assignment; zeros if its block never ran).
#[derive(Clone, Debug, PartialEq)]
pub struct TraceValue {
    /// The binding's name.
    pub name: String,
    /// Its type.
    pub ty: Type,
    /// Its words, in the ABI's layout.
    pub words: Vec<u32>,
    /// Whether the binding's block ran for this element; `words` holds
    /// meaning only when it did. A loop body runs when its loop does.
    pub ran: bool,
}

/// One element of a Kernel run on the interpreter, every binding shown.
#[derive(Clone, Debug, PartialEq)]
pub struct Trace {
    /// The element's coordinates (a scan's step is `[step, 0, 0]`).
    pub at: [u32; 3],
    /// Every `let` of the entry, in source order.
    pub bindings: Vec<TraceValue>,
    /// Every call the element made, in the order the calls returned
    /// (a call inside another comes first), the first `CALL_RECORDS`.
    pub calls: Vec<TraceCall>,
    /// Whether the element made more calls than were recorded.
    pub calls_truncated: bool,
    /// The entry's results: the outputs' elements (a scan's next carried
    /// value, a scatter's pair, a reduction's element before the fold).
    pub results: Vec<TraceValue>,
    /// Where the element failed, when it did: the bindings above hold
    /// what ran before it, and none after.
    pub fault: Option<TraceFault>,
}

/// Where a traced element failed.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceFault {
    /// What went wrong, as the run reports it.
    pub message: String,
    /// The function it failed in, `None` the entry.
    pub function: Option<String>,
    /// The failing value in that function.
    pub value: Option<entelechy_ir::ValueId>,
    /// In the entry, the value it failed under: the failing one, or the
    /// call it failed in.
    pub entry_value: Option<entelechy_ir::ValueId>,
    /// That function's bindings as the failure left them, when it failed
    /// in a function.
    pub bindings: Vec<TraceValue>,
}

/// The loop whose spend `function` wrote at frame word `word`.
fn loop_spent_at(
    function: &entelechy_ir::Function,
    locs: &[(usize, usize)],
    word: usize,
) -> Option<entelechy_ir::ValueId> {
    fn walk(
        body: &[entelechy_ir::Instruction],
        locs: &[(usize, usize)],
        word: usize,
    ) -> Option<entelechy_ir::ValueId> {
        body.iter().find_map(|instruction| match instruction.op {
            entelechy_ir::Op::Spend(looped) if locs[instruction.id.0 as usize].0 == word => {
                Some(looped)
            }
            _ => instruction
                .op
                .blocks()
                .into_iter()
                .find_map(|block| walk(&block.body, locs, word)),
        })
    }
    walk(&function.body, locs, word)
}

/// The first value of `function` an operation that can fail wrote at
/// frame word `word`, by `locs` (each value's word and width).
fn value_written_at(
    function: &entelechy_ir::Function,
    locs: &[(usize, usize)],
    word: usize,
) -> Option<entelechy_ir::ValueId> {
    fn walk(
        body: &[entelechy_ir::Instruction],
        locs: &[(usize, usize)],
        word: usize,
        any: bool,
    ) -> Option<entelechy_ir::ValueId> {
        for instruction in body {
            let (at, n) = locs[instruction.id.0 as usize];
            let fails = matches!(
                instruction.op,
                entelechy_ir::Op::Binary(..)
                    | entelechy_ir::Op::Unary(..)
                    | entelechy_ir::Op::Intrinsic(..)
                    | entelechy_ir::Op::Call(..)
                    | entelechy_ir::Op::Loop { .. }
            );
            if (any || fails) && n > 0 && at == word {
                return Some(instruction.id);
            }
            for block in instruction.op.blocks() {
                if let Some(found) = walk(&block.body, locs, word, any) {
                    return Some(found);
                }
            }
        }
        None
    }
    walk(&function.body, locs, word, false).or_else(|| walk(&function.body, locs, word, true))
}

/// One call the traced element made, as it returned.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceCall {
    /// The function's name (a template instance names its Field, as in
    /// `store<ram>`).
    pub function: String,
    /// Every `let` of the function, in source order.
    pub bindings: Vec<TraceValue>,
    /// The function's results.
    pub results: Vec<TraceValue>,
}

/// How many calls a trace records.
pub const CALL_RECORDS: usize = eval::CALL_RECORDS;

/// Which entry values ran for the element just run, read from the
/// conditions: an `if` branch ran when its `if` ran and chose it; a loop
/// value ran when its loop did (showing the last iteration).
fn ran_values(
    entry: &entelechy_ir::Function,
    value: &dyn Fn(entelechy_ir::ValueId) -> Vec<u32>,
    failed: Option<entelechy_ir::ValueId>,
) -> Vec<bool> {
    struct Walk<'a> {
        value: &'a dyn Fn(entelechy_ir::ValueId) -> Vec<u32>,
        failed: Option<entelechy_ir::ValueId>,
        stopped: bool,
    }
    fn walk(
        instructions: &[entelechy_ir::Instruction],
        live: bool,
        w: &mut Walk<'_>,
        ran: &mut [bool],
    ) {
        for instruction in instructions {
            // The failing value did not come to be, nor anything after it.
            w.stopped |= w.failed == Some(instruction.id);
            let live = live && !w.stopped;
            ran[instruction.id.0 as usize] = live;
            let value = w.value;
            match &instruction.op {
                entelechy_ir::Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    let chosen = live && value(*condition)[0] != 0;
                    walk(&then.body, chosen, w, ran);
                    walk(&otherwise.body, live && !chosen, w, ran);
                }
                entelechy_ir::Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    let matched = value(*scrutinee)[0] as i32;
                    let mut taken = false;
                    for (values, block) in arms {
                        let chosen = live && !taken && values.contains(&matched);
                        taken |= chosen;
                        walk(&block.body, chosen, w, ran);
                    }
                    walk(&otherwise.body, live && !taken, w, ran);
                }
                entelechy_ir::Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, live, w, ran);
                    }
                    walk(&body.body, live, w, ran);
                }
                entelechy_ir::Op::Each { body, .. } => walk(&body.body, live, w, ran),
                _ => {}
            }
        }
    }
    let mut ran = vec![false; entry.slot_count()];
    let mut w = Walk {
        value,
        failed,
        stopped: false,
    };
    walk(&entry.body, true, &mut w, &mut ran);
    ran
}

/// The named values of a function from its frame, the ones whose block
/// ran marked, and its results.
fn values_of(
    function: &entelechy_ir::Function,
    value: &dyn Fn(entelechy_ir::ValueId) -> Vec<u32>,
    result: &dyn Fn(usize) -> Vec<u32>,
    result_names: Vec<String>,
    failed: Option<entelechy_ir::ValueId>,
) -> (Vec<TraceValue>, Vec<TraceValue>) {
    let ran = ran_values(function, value, failed);
    let bindings = function
        .bindings
        .iter()
        .map(|(id, name)| TraceValue {
            name: name.clone(),
            ty: function.type_of(*id).unwrap_or(Type::Float),
            words: value(*id),
            ran: ran[id.0 as usize],
        })
        .collect();
    let results = function
        .results
        .iter()
        .zip(result_names)
        .enumerate()
        .map(|(index, (id, name))| TraceValue {
            name,
            ty: function.type_of(*id).unwrap_or(Type::Float),
            words: result(index),
            ran: failed.is_none(),
        })
        .collect();
    (bindings, results)
}

/// Runs one element of `module` on the interpreter and returns every
/// binding of its entry: element `at` of a map (or of the map part of a
/// reduction or a scatter), or step `at[0]` of a scan after running the
/// earlier steps. The scope's Fields are only read.
pub fn trace(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    at: [u32; 3],
) -> Result<Trace, ExecutionError> {
    check_invocation(module, invocation, scope)?;
    let mode = module.manifest.mode;
    let domain = match mode {
        Mode::Map | Mode::Group { .. } => invocation.domain,
        Mode::Reduce | Mode::Prefix { .. } => scope.size(invocation.inputs[0])?,
        Mode::Scan => check_scan(invocation, scope)?,
        Mode::Scatter { base, .. } => {
            check_scatter(invocation, scope, base as usize)?;
            invocation.domain
        }
    };
    let inside = if mode == Mode::Scan {
        (at[0] as usize) < domain.count()
    } else {
        at[0] < domain.width() && at[1] < domain.height() && at[2] < domain.depth()
    };
    if !inside {
        return Err(ExecutionError::Device(format!(
            "element ({}, {}, {}) is outside the domain {domain}",
            at[0], at[1], at[2]
        )));
    }
    let program = eval::Program::new(module, invocation, scope, domain)?;
    let mut machine = eval::Machine::new(&program);
    let words: usize = module
        .manifest
        .outputs
        .iter()
        .take(if mode == Mode::Scan { 1 } else { usize::MAX })
        .map(|port| port.ty.schema.value_words())
        .sum();
    let stride = match mode {
        Mode::Scatter { .. } => {
            address_words(domain) + module.manifest.outputs[0].ty.schema.value_words()
        }
        _ => words,
    };
    let mut out = vec![0u32; stride];
    let mut ran = Ok(());
    if mode == Mode::Scan {
        let mut carried = vec![0u32; stride];
        machine.run_init_words(&mut carried)?;
        for step in 0..=at[0] {
            if step == at[0] {
                machine.clear_entry_frame();
                machine.record_calls();
                ran = machine.run_scan_step(step, &carried, &mut out);
            } else {
                machine.run_scan_step(step, &carried, &mut out)?;
            }
            carried.copy_from_slice(&out);
        }
    } else if let Mode::Group { size } = mode {
        // A lane takes what its group gave: the group runs its lanes
        // first, keeping what they gave at each exchange, then the lane
        // is traced taking it.
        let size = size as usize;
        let element = linear(at, domain);
        let lowered = Code::of(module).map_err(ExecutionError::Device)?;
        let group = eval::Program::with(lowered.0, module, invocation, scope, domain)?;
        let mut lanes = eval::Machine::new(&group);
        let mut slots = vec![0u32; size * stride];
        let given = lanes
            .run_group(element / size, size, &mut slots, true)?
            .expect("kept");
        drop(lanes);
        drop(group);
        machine.answers = Some(eval::Answers {
            size,
            lane: element % size,
            given,
            passed: 0,
        });
        machine.record_calls();
        ran = machine.run_entry_words(at, &mut out);
    } else {
        machine.record_calls();
        ran = machine.run_entry_words(at, &mut out);
    }
    // An element's own failure is traced up to it; any other stops here.
    let failed = match ran {
        Ok(()) => None,
        Err(error @ (ExecutionError::Arithmetic { .. } | ExecutionError::Exhausted { .. })) => {
            let exhausted = matches!(error, ExecutionError::Exhausted { .. });
            Some((error.to_string(), exhausted))
        }
        Err(error) => return Err(error),
    };
    let entry = &module.entry;
    let names: Vec<String> =
        if mode == Mode::Map && module.manifest.outputs.len() == entry.results.len() {
            module
                .manifest
                .outputs
                .iter()
                .map(|port| port.name.clone())
                .collect()
        } else {
            entry.results.iter().map(|_| "result".to_owned()).collect()
        };
    // The failing value: in the entry, or in the innermost function, and
    // then the entry's call it failed under.
    let mut fault = None;
    let mut entry_failed = None;
    if let Some((message, exhausted)) = failed {
        let failure = machine.failure();
        let entry_locs = machine.locs(None);
        let (function, value, bindings) = match &failure {
            Some(f) if f.function.is_some() => {
                let index = f.function.unwrap();
                let function = &module.functions[index];
                let value = value_written_at(function, &machine.locs(Some(index)), f.word);
                let (bindings, _) = values_of(
                    function,
                    &|id| machine.function_value(index, &f.frame, id),
                    &|index_| machine.function_result(index, &f.frame, index_),
                    Vec::new(),
                    value,
                );
                entry_failed = f
                    .entry_word
                    .and_then(|word| value_written_at(entry, &entry_locs, word));
                (Some(function.name.clone()), value, bindings)
            }
            Some(f) => {
                entry_failed = value_written_at(entry, &entry_locs, f.word);
                (None, entry_failed, Vec::new())
            }
            // An element past its budget as it ends is named by the loop
            // that ran the most iterations; everything it wrote ran.
            None if exhausted => match machine.most_spent() {
                Some((index, word)) => {
                    let function = index.map_or(entry, |index| &module.functions[index]);
                    (
                        index.map(|_| function.name.clone()),
                        loop_spent_at(function, &machine.locs(index), word),
                        Vec::new(),
                    )
                }
                None => (None, None, Vec::new()),
            },
            None => (None, None, Vec::new()),
        };
        fault = Some(TraceFault {
            message,
            function,
            value,
            entry_value: entry_failed,
            bindings,
        });
    }
    let (bindings, results) = values_of(
        entry,
        &|id| machine.entry_value(id),
        &|index| machine.entry_result(index),
        names,
        entry_failed,
    );
    let (records, calls_truncated) = machine.take_calls();
    let calls = records
        .iter()
        .map(|record| {
            let function = &module.functions[record.function];
            let (bindings, results) = values_of(
                function,
                &|id| machine.function_value(record.function, &record.frame, id),
                &|index| machine.function_result(record.function, &record.frame, index),
                function
                    .results
                    .iter()
                    .map(|_| "result".to_owned())
                    .collect(),
                None,
            );
            TraceCall {
                function: function.name.clone(),
                bindings,
                results,
            }
        })
        .collect();
    Ok(Trace {
        at: if mode == Mode::Scan {
            [at[0], 0, 0]
        } else {
            at
        },
        bindings,
        calls,
        calls_truncated,
        results,
        fault,
    })
}

/// A scan: the entry once per element of the first input, in order, each
/// step reading the previous step's result and writing its own, on one
/// thread.
fn scan(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = check_scan(invocation, scope)?;
    let count = domain.count();
    let program = eval::Program::with(code.clone(), module, invocation, scope, domain)?;
    let mut machine = eval::Machine::new(&program);
    let mut carried = vec![0u32; stride];
    machine.run_init_words(&mut carried)?;
    let mut words = vec![0u32; count * stride];
    for index in 0..count {
        let out = &mut words[index * stride..(index + 1) * stride];
        machine.run_scan_step(index as u32, &carried, out)?;
        carried.copy_from_slice(out);
    }
    // The final memories are the outputs after the first, in manifest
    // order.
    machine.finish()?;
    program.outcome()?;
    let memories = machine.take_memories();
    drop(machine);
    drop(program);
    let port = &module.manifest.outputs[0];
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    let mut outputs = vec![output];
    for (input, words) in memories {
        let port = &module.manifest.inputs[input];
        let extent = match scope.size(invocation.inputs[input]) {
            Ok(extent) => extent,
            Err(error) => {
                release_all(scope, &outputs);
                return Err(error.into());
            }
        };
        let created = scope
            .create(port.ty.schema.clone(), extent)
            .and_then(|handle| scope.write_storage(handle, &words).map(|()| handle));
        match created {
            Ok(handle) => outputs.push(handle),
            Err(error) => {
                release_all(scope, &outputs);
                return Err(error.into());
            }
        }
    }
    Ok(outputs)
}

/// A scatter: the entry over the domain, then every address and value
/// applied to a copy of the target in element order, which every backend
/// matches, floats included.
fn scatter(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
    base: usize,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let target = check_scatter(invocation, scope, base)?;
    let sources = invocation.domain;
    let address = address_words(target);
    let program = eval::Program::with(code.clone(), module, invocation, scope, sources)?;
    let held = entelechy_rules::Held::new(scope.room());
    let (pairs, sources_of) = compute_pairs(&program, address + stride, &held)?;
    program.outcome()?;
    drop(program);
    let mut words = scope.values(invocation.inputs[base])?.into_owned();
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars();
    // A sum is each element's whole total in any order; an overflow
    // names the element whose total leaves the range.
    if combine == Combine::Sum {
        if !held.take(entelechy_rules::Sums::held_words(scalars, target.count())) {
            return Err(held.over());
        }
        let mut sums = entelechy_rules::Sums::new(scalars, &words);
        for pair in pairs.chunks_exact(address + stride) {
            if let Some(at) = scatter_index(target, &pair[..address]) {
                sums.add(&mut words, at, &pair[address..]);
            }
        }
        sums.finish(&mut words)
            .map_err(|(element, error)| ExecutionError::Arithmetic {
                x: (element % target.width() as usize) as u32,
                y: (element / target.width() as usize) as u32,
                error,
            })?;
        let output = scope.create(port.ty.schema.clone(), target)?;
        if let Err(error) = scope.write_values(output, &words) {
            let _ = scope.release(output);
            return Err(error.into());
        }
        return Ok(vec![output]);
    }
    for (pair, index) in pairs.chunks_exact(address + stride).zip(sources_of) {
        let index = index as usize;
        let Some(at) = scatter_index(target, &pair[..address]) else {
            continue;
        };
        combine_words(
            combine,
            scalars,
            &mut words[at * stride..(at + 1) * stride],
            &pair[address..],
        )
        .map_err(|error| ExecutionError::Arithmetic {
            x: (index % sources.width() as usize) as u32,
            y: (index / sources.width() as usize) as u32,
            error,
        })?;
    }
    let output = scope.create(port.ty.schema.clone(), target)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// A scatter's pairs' words, and the element each pair came from.
type Pairs = (Vec<u32>, Vec<u32>);

/// A scatter's pairs, in the order the rule applies them: element by
/// element, its emitted pairs then its returned one, each with its
/// element. Threads' runs join in run order, as one thread's would.
/// Every pair and its element word are charged to the Host's budget: the
/// returned pairs before the run, emitted ones as they come.
fn compute_pairs(
    program: &eval::Program<'_>,
    pair_words: usize,
    held: &entelechy_rules::Held,
) -> Result<Pairs, ExecutionError> {
    let domain = program.domain;
    let width = domain.width() as usize;
    let height = domain.height();
    let element_count = width * height as usize * domain.depth() as usize;
    let threads = entelechy_rules::workers(element_count);
    let per_thread = element_count.div_ceil(threads).max(1);
    if !held.take(element_count * (pair_words + 1)) {
        return Err(held.over());
    }
    let run = |first: usize| -> Result<Pairs, ExecutionError> {
        let mut machine = eval::Machine::new(program);
        machine.hold(held);
        let mut pairs = Vec::new();
        let mut sources = Vec::new();
        let mut slot = vec![0u32; pair_words];
        for element in first..(first + per_thread).min(element_count) {
            let x = (element % width.max(1)) as u32;
            let row = (element / width.max(1)) as u32;
            let (y, z) = (row % height, row / height);
            machine.run_entry_words([x, y, z], &mut slot)?;
            let emitted = machine.take_emitted();
            sources.extend(std::iter::repeat_n(
                element as u32,
                emitted.len() / pair_words + 1,
            ));
            pairs.extend(emitted);
            pairs.extend_from_slice(&slot);
        }
        machine.finish()?;
        Ok((pairs, sources))
    };
    if threads == 1 {
        return run(0);
    }
    let runs: Vec<Result<Pairs, ExecutionError>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|index| {
                let run = &run;
                s.spawn(move || run(index * per_thread))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("scatter worker panicked"))
            .collect()
    });
    let (mut pairs, mut sources) = (Vec::new(), Vec::new());
    for run in runs {
        let (p, s) = run?;
        pairs.extend(p);
        sources.extend(s);
    }
    Ok((pairs, sources))
}

/// Runs the entry over the program's domain on as many threads as pay
/// off. Returns every output's words per element, outputs concatenated.
fn compute(program: &eval::Program<'_>, strides: &[usize]) -> Result<Vec<u32>, ExecutionError> {
    let domain = program.domain;
    let width = domain.width() as usize;
    let height = domain.height();
    // Rows of every layer, one after another: the third axis is outermost.
    let rows = height as usize * domain.depth() as usize;
    let stride: usize = strides.iter().sum();
    let row_words = width * stride;
    let mut results = vec![0u32; rows * row_words];
    let element_count = width * rows;
    // Split by elements: a `Rows` domain is a single row.
    let threads = entelechy_rules::workers(element_count);
    let per_thread = element_count.div_ceil(threads).max(1);
    let run_elements = |first: usize, chunk: &mut [u32]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(program);
        for (offset, slot) in chunk.chunks_exact_mut(stride.max(1)).enumerate() {
            let element = first + offset;
            let x = (element % width.max(1)) as u32;
            let row = (element / width.max(1)) as u32;
            let (y, z) = (row % height, row / height);
            machine.run_entry_words([x, y, z], slot)?;
        }
        machine.finish()
    };
    if threads == 1 {
        run_elements(0, &mut results)?;
        return Ok(results);
    }
    let chunk_len = per_thread * stride.max(1);
    let outcomes: Vec<Result<(), ExecutionError>> = std::thread::scope(|s| {
        let handles: Vec<_> = results
            .chunks_mut(chunk_len)
            .enumerate()
            .map(|(index, chunk)| {
                let run_elements = &run_elements;
                s.spawn(move || run_elements(index * per_thread, chunk))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("row worker panicked"))
            .collect()
    });
    for outcome in outcomes {
        outcome?;
    }
    Ok(results)
}

/// The linear index of the element at `at` in `domain`, `x` fastest: what
/// `element()` reads, past the end too.
fn linear(at: [u32; 3], domain: Domain) -> usize {
    let [x, y, z] = at.map(|c| c as usize);
    (z * domain.height() as usize + y) * domain.width() as usize + x
}

/// Every element's words, `strides` an output, run a group of `size` at
/// a time, the entry running its lanes ([`entelechy_ir::lanes`]); a group
/// past the domain's end runs whole and its lanes past the end write
/// nothing.
fn compute_group(
    program: &eval::Program<'_>,
    strides: &[usize],
    size: usize,
) -> Result<Vec<u32>, ExecutionError> {
    let domain = program.domain;
    let count = domain.count();
    let stride: usize = strides.iter().sum();
    let groups = count.div_ceil(size);
    let mut results = vec![0u32; groups * size * stride];
    let threads = entelechy_rules::workers(count);
    let per_thread = groups.div_ceil(threads).max(1);
    let run_groups = |first: usize, chunk: &mut [u32]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(program);
        // Several outputs come out an array of lanes each, laid here and
        // then element after element.
        let mut lanes = vec![0u32; if strides.len() > 1 { size * stride } else { 0 }];
        for (offset, slots) in chunk.chunks_exact_mut(size * stride.max(1)).enumerate() {
            if strides.len() == 1 {
                machine.run_group(first + offset, size, slots, false)?;
                continue;
            }
            machine.run_group(first + offset, size, &mut lanes, false)?;
            let mut base = 0;
            let mut within = 0;
            for words in strides {
                for lane in 0..size {
                    let from = base + lane * words;
                    let to = lane * stride + within;
                    slots[to..to + words].copy_from_slice(&lanes[from..from + words]);
                }
                base += size * words;
                within += words;
            }
        }
        machine.finish()
    };
    if threads == 1 || stride == 0 {
        run_groups(0, &mut results)?;
    } else {
        let chunk_len = per_thread * size * stride;
        let outcomes: Vec<Result<(), ExecutionError>> = std::thread::scope(|s| {
            let handles: Vec<_> = results
                .chunks_mut(chunk_len)
                .enumerate()
                .map(|(index, chunk)| {
                    let run_groups = &run_groups;
                    s.spawn(move || run_groups(index * per_thread, chunk))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("group worker panicked"))
                .collect()
        });
        for outcome in outcomes {
            outcome?;
        }
    }
    results.truncate(count * stride);
    Ok(results)
}

fn release_all(scope: &mut Scope<'_>, outputs: &[Handle]) {
    for output in outputs {
        // Created through this scope, so release is granted; an error
        // means the handle is already gone.
        let _ = scope.release(*output);
    }
}

/// The interpreter's word code, compiled once when the module is
/// prepared.
struct CpuCode(Code);

impl Prepared for CpuCode {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &'static str {
        "cpu"
    }

    /// The reference walks each element in order: the same words each run.
    fn repeats_exactly(&self) -> bool {
        true
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        Ok(Box::new(CpuCode(Code::of(module)?)))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let code = prepared.as_any().downcast_ref::<CpuCode>().map(|c| &c.0);
        CpuBackend::execute_with(self, code, module, invocation, scope)
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        match op {
            NativeOp::Rasterize => entelechy_rules::raster::rasterize(invocation, scope),
            NativeOp::RasterizeCutout => {
                entelechy_rules::raster::rasterize_cutout(invocation, scope)
            }
            NativeOp::Cover => entelechy_rules::raster::cover(invocation, scope),
            NativeOp::RasterizeCanvas => {
                entelechy_rules::raster::rasterize_canvas(invocation, scope)
            }
            NativeOp::RasterizeCutoutCanvas => {
                entelechy_rules::raster::rasterize_cutout_canvas(invocation, scope)
            }
            NativeOp::Trace => entelechy_rules::trace::trace(invocation, scope),
            NativeOp::TraceInstances => entelechy_rules::trace::trace_instances(invocation, scope),
        }
    }
}
