#![warn(missing_docs)]
//! CPU backend.
//!
//! Compiles a [`Validated`] module to word-addressed code — once, into a
//! [`Code`] a caller keeps — and runs it once per output element. The backend is trusted Rust, but the Kernel it runs
//! is not: every Host access goes through the capability-gated [`Scope`],
//! and the backend creates and releases the outputs it allocates so a
//! failed execution leaks nothing.
//!
//! Execution has two phases. The compute phase borrows the inputs through
//! the scope once and runs rows in parallel; the write phase stores results
//! sequentially. Results are identical to a single-threaded run, and the
//! first failing pixel in row-major order is the one reported.

mod eval;

use std::any::Any;

use orla_abi::{Domain, Handle, Invocation, Scope, Type};
use orla_ir::{Combine, Mode, Validated};

use orla_rules::{Backend, NativeOp, Prepared};

pub use orla_rules::{
    ArithmeticError, ExecutionError, address_words, check_invocation, check_scan, check_scatter,
    combine_words, float_order_key, scatter_index,
};

/// Elements per thread below which the backend stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// The CPU backend. Stateless; one instance can execute any module.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

/// A module compiled to the interpreter's word code. Compiling a
/// Kernel of a hundred statements takes about eighty microseconds,
/// which a frame should not pay for every node it runs, so a backend
/// compiles once and binds an invocation to it.
#[derive(Clone)]
pub struct Code(std::sync::Arc<eval::Compiled>);

impl Code {
    /// Compiles a module.
    pub fn of(module: &Validated) -> Self {
        Self(std::sync::Arc::new(eval::Program::compile(module)))
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
            None => Code::of(module).0,
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
        // The words are computed here before the Host holds them: asked
        // first, so a domain past the budget costs nothing to refuse.
        let fields: Vec<_> = module
            .manifest
            .outputs
            .iter()
            .map(|port| (&port.ty.schema, invocation.domain))
            .collect();
        scope.can_hold(&fields)?;
        let program = eval::Program::with(code, module, invocation, scope, invocation.domain)?;
        let results = compute(&program, &strides)?;
        program.outcome()?;
        drop(program);
        let domain = invocation.domain;
        let stride: usize = strides.iter().sum();
        let count = domain.count();
        // One output is the common case, and then what the elements
        // wrote is already the Field's words in order: it goes in
        // whole, rather than being gathered into a second buffer first.
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
/// fold with `merge` in a fixed tree, so the result does not depend on
/// how many threads ran the map. One `Rows` element comes out.
fn reduce(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
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
/// is order-free, so this walk is one grouping of many and every
/// backend's answer is this one.
fn prefix(
    code: &std::sync::Arc<eval::Compiled>,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
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
            orla_rules::combine_words(combine, scalars, &mut running, &values[at..at + stride])
                .map_err(|error| ExecutionError::Arithmetic {
                    x: element as u32,
                    y: 0,
                    error,
                })?;
            values[at..at + stride].copy_from_slice(&running);
        }
    }
    // The output is the input's own count, which is what the walk just
    // wrote, element for element.
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
    /// Whether the binding's block ran for this element: a `let` in the
    /// branch of an `if` not taken did not, and its words mean nothing.
    /// A `let` in a loop body counts as run when the loop ran.
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

/// Which values of the entry were computed for the element just run,
/// read back from the conditions: a value in the branch of an `if` ran
/// when the `if` ran and its condition chose that branch; a value in a
/// loop ran when the loop did (its last iteration's value is shown).
fn ran_values(
    entry: &orla_ir::Function,
    value: &dyn Fn(orla_ir::ValueId) -> Vec<u32>,
) -> Vec<bool> {
    fn walk(
        instructions: &[orla_ir::Instruction],
        live: bool,
        value: &dyn Fn(orla_ir::ValueId) -> Vec<u32>,
        ran: &mut [bool],
    ) {
        for instruction in instructions {
            ran[instruction.id.0 as usize] = live;
            match &instruction.op {
                orla_ir::Op::If {
                    condition,
                    then,
                    otherwise,
                } => {
                    let chosen = live && value(*condition)[0] != 0;
                    walk(&then.body, chosen, value, ran);
                    walk(&otherwise.body, live && !chosen, value, ran);
                }
                orla_ir::Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    let matched = value(*scrutinee)[0] as i32;
                    let mut taken = false;
                    for (values, block) in arms {
                        let chosen = live && !taken && values.contains(&matched);
                        taken |= chosen;
                        walk(&block.body, chosen, value, ran);
                    }
                    walk(&otherwise.body, live && !taken, value, ran);
                }
                orla_ir::Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, live, value, ran);
                    }
                    walk(&body.body, live, value, ran);
                }
                _ => {}
            }
        }
    }
    let mut ran = vec![false; entry.slot_count()];
    walk(&entry.body, true, value, &mut ran);
    ran
}

/// The named values of a function from its frame, the ones whose block
/// ran marked, and its results.
fn values_of(
    function: &orla_ir::Function,
    value: &dyn Fn(orla_ir::ValueId) -> Vec<u32>,
    result: &dyn Fn(usize) -> Vec<u32>,
    result_names: Vec<String>,
) -> (Vec<TraceValue>, Vec<TraceValue>) {
    let ran = ran_values(function, value);
    let bindings = function
        .bindings
        .iter()
        .map(|(id, name)| TraceValue {
            name: name.clone(),
            ty: type_of(function, *id).unwrap_or(Type::Float),
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
            ty: type_of(function, *id).unwrap_or(Type::Float),
            words: result(index),
            ran: true,
        })
        .collect();
    (bindings, results)
}

/// The type of a value of a function, by id, nested blocks included.
fn type_of(function: &orla_ir::Function, id: orla_ir::ValueId) -> Option<Type> {
    fn find(instructions: &[orla_ir::Instruction], id: orla_ir::ValueId) -> Option<Type> {
        for instruction in instructions {
            if instruction.id == id {
                return Some(instruction.ty);
            }
            let nested = match &instruction.op {
                orla_ir::Op::If {
                    then, otherwise, ..
                } => find(&then.body, id).or_else(|| find(&otherwise.body, id)),
                orla_ir::Op::Match {
                    arms, otherwise, ..
                } => arms
                    .iter()
                    .find_map(|(_, block)| find(&block.body, id))
                    .or_else(|| find(&otherwise.body, id)),
                orla_ir::Op::Loop { until, body, .. } => until
                    .as_ref()
                    .and_then(|block| find(&block.body, id))
                    .or_else(|| find(&body.body, id)),
                _ => None,
            };
            if nested.is_some() {
                return nested;
            }
        }
        None
    }
    find(&function.body, id)
}

/// Runs one element of `module` on the interpreter and returns every
/// binding of its entry: element `at` of a map (or of the map part of a
/// reduction or a scatter), or step `at[0]` of a scan, the earlier steps
/// run first so the memories and the carried value are as that step saw
/// them. Nothing is written back; the scope's Fields are only read.
pub fn trace(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    at: [u32; 3],
) -> Result<Trace, ExecutionError> {
    check_invocation(module, invocation, scope)?;
    let mode = module.manifest.mode;
    let domain = match mode {
        Mode::Map => invocation.domain,
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
    if mode == Mode::Scan {
        let mut carried = vec![0u32; stride];
        machine.run_init_words(&mut carried)?;
        for step in 0..=at[0] {
            if step == at[0] {
                machine.clear_entry_frame();
                machine.record_calls();
            }
            machine.run_scan_step(step, &carried, &mut out)?;
            carried.copy_from_slice(&out);
        }
    } else {
        machine.record_calls();
        machine.run_entry_words(at, &mut out)?;
    }
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
    let (bindings, results) = values_of(
        entry,
        &|id| machine.entry_value(id),
        &|index| machine.entry_result(index),
        names,
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
    })
}

/// A scan: the entry once per element of the first input, in order, each
/// step reading the previous step's result and writing its own. Ordered
/// work runs on one thread; that is its nature, not a limitation of the
/// backend.
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
    // The memories as the steps left them become the outputs after the
    // first, in the manifest's order.
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
/// applied to a copy of the target, one after another in element order.
/// The order is the rule, so the result is the same on every backend,
/// floats included.
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
    let held = orla_rules::Held::new(scope.room());
    let (pairs, sources_of) = compute_pairs(&program, address + stride, &held)?;
    program.outcome()?;
    drop(program);
    let mut words = scope.values(invocation.inputs[base])?.into_owned();
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars();
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
/// element, the pairs each emits and then the one it returns, with the
/// element each came from. Threads take runs of elements and their pairs
/// are joined in run order, so the order is the one thread's would be.
/// Every pair is held against the Host's room with the word naming its
/// element: the returned ones before the run, the emitted as they come.
fn compute_pairs(
    program: &eval::Program<'_>,
    pair_words: usize,
    held: &orla_rules::Held,
) -> Result<Pairs, ExecutionError> {
    let domain = program.domain;
    let width = domain.width() as usize;
    let height = domain.height();
    let element_count = width * height as usize * domain.depth() as usize;
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(element_count.max(1))
        .min(element_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
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

/// Runs the entry point over every element of the program's domain, on
/// as many threads as pay off. Returns every output's words per element,
/// outputs concatenated.
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
    // Split by elements and not by rows: a `Rows` domain is one row of
    // a million, and splitting that by rows is one thread.
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(element_count.max(1))
        .min(element_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
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

fn release_all(scope: &mut Scope<'_>, outputs: &[Handle]) {
    for output in outputs {
        // The output was created through this scope, so release cannot be
        // denied; a Host failure here would mean the handle is already gone.
        let _ = scope.release(*output);
    }
}

/// The interpreter's word code, compiled once when the module is
/// prepared. A Kernel of a hundred statements takes tens of
/// microseconds to compile, which is more than most nodes take to run,
/// so a frame must not pay it per execution.
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

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        Ok(Box::new(CpuCode(Code::of(module))))
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
            NativeOp::Rasterize => orla_rules::raster::rasterize(invocation, scope),
            NativeOp::RasterizeCutout => orla_rules::raster::rasterize_cutout(invocation, scope),
        }
    }
}
