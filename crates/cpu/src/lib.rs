#![warn(missing_docs)]
//! CPU backend.
//!
//! Compiles a [`Validated`] module to word-addressed code and runs it once
//! per output element. The backend is trusted Rust, but the Kernel it runs
//! is not: every Host access goes through the capability-gated [`Scope`],
//! and the backend creates and releases the outputs it allocates so a
//! failed execution leaks nothing.
//!
//! Execution has two phases. The compute phase borrows the inputs through
//! the scope once and runs rows in parallel; the write phase stores results
//! sequentially. Results are identical to a single-threaded run, and the
//! first failing pixel in row-major order is the one reported.

mod eval;
pub mod raster;

use std::fmt;

use orla_abi::{AbiError, Domain, Handle, Invocation, Scalar, Scope, Shape, Type};
use orla_ir::{Combine, Mode, Validated};

pub use eval::ArithmeticError;

/// Why an execution failed. The Kernel version stays valid; only this run
/// produced nothing.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ExecutionError {
    /// The invocation supplied the wrong number of inputs.
    InputCount {
        /// Inputs the manifest declares.
        expected: usize,
        /// Inputs supplied.
        actual: usize,
    },
    /// The invocation supplied the wrong number of parameters.
    ParamCount {
        /// Parameters the manifest declares.
        expected: usize,
        /// Parameters supplied.
        actual: usize,
    },
    /// A parameter has the wrong type.
    ParamType {
        /// Parameter name.
        name: String,
        /// Declared type.
        expected: Type,
        /// Supplied type.
        actual: Type,
    },
    /// The output domain has no elements.
    EmptyDomain,
    /// The output domain's shape is not the output port's.
    DomainShape {
        /// The port's shape.
        expected: Shape,
        /// The invocation's shape.
        actual: Shape,
    },
    /// A Host call was denied or rejected.
    Abi(AbiError),
    /// The backend device failed; the message is backend specific.
    Device(String),
    /// Integer arithmetic failed at the given element.
    Arithmetic {
        /// Element column.
        x: u32,
        /// Element row, rows of every layer counted through.
        y: u32,
        /// What failed.
        error: ArithmeticError,
    },
}

impl fmt::Display for ExecutionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InputCount { expected, actual } => {
                write!(f, "kernel expects {expected} input(s), received {actual}")
            }
            Self::ParamCount { expected, actual } => {
                write!(
                    f,
                    "kernel expects {expected} parameter(s), received {actual}"
                )
            }
            Self::ParamType {
                name,
                expected,
                actual,
            } => write!(
                f,
                "parameter `{name}` expects {expected}, received {actual}"
            ),
            Self::EmptyDomain => write!(f, "output domain has no elements"),
            Self::DomainShape { expected, actual } => {
                write!(f, "output is a {expected} but the domain is a {actual}")
            }
            Self::Abi(error) => write!(f, "{error}"),
            Self::Device(message) => write!(f, "device failed: {message}"),
            Self::Arithmetic { x, y, error } => write!(f, "at element ({x}, {y}): {error}"),
        }
    }
}

impl std::error::Error for ExecutionError {}

impl From<AbiError> for ExecutionError {
    fn from(error: AbiError) -> Self {
        Self::Abi(error)
    }
}

/// Elements per thread below which the backend stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// The CPU backend. Stateless; one instance can execute any module.
#[derive(Clone, Copy, Debug, Default)]
pub struct CpuBackend;

impl CpuBackend {
    /// Executes `module` over `invocation.domain`, returning one handle per
    /// manifest output. The handles are owned by the caller's Host.
    pub fn execute(
        &self,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        check_invocation(module, invocation)?;
        let strides = output_strides(module);
        if module.manifest.mode == Mode::Reduce {
            return reduce(module, invocation, scope, strides[0]);
        }
        if module.manifest.mode == Mode::Scan {
            return scan(module, invocation, scope, strides[0]);
        }
        if let Mode::Scatter { base, combine } = module.manifest.mode {
            return scatter(
                module,
                invocation,
                scope,
                strides[0],
                base as usize,
                combine,
            );
        }
        let program = eval::Program::new(module, invocation, scope, invocation.domain)?;
        let results = compute(&program, &strides)?;
        drop(program);
        let domain = invocation.domain;
        let stride: usize = strides.iter().sum();
        let count = domain.count();
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
                .and_then(|handle| scope.write_all(handle, &own).map(|()| handle));
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
        .map(|port| port.ty.schema.words())
        .collect()
}

/// A reduction: the entry over the first input's domain, then a pairwise
/// fold with `merge` in a fixed tree, so the result does not depend on
/// how many threads ran the map. One `Rows` element comes out.
fn reduce(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    let program = eval::Program::new(module, invocation, scope, domain)?;
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
    drop(machine);
    drop(program);
    let port = &module.manifest.outputs[0];
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_all(output, &level) {
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
    /// The entry's results: the outputs' elements (a scan's next carried
    /// value, a scatter's pair, a reduction's element before the fold).
    pub results: Vec<TraceValue>,
}

/// Which values of the entry were computed for the element just run,
/// read back from the conditions: a value in the branch of an `if` ran
/// when the `if` ran and its condition chose that branch; a value in a
/// loop ran when the loop did (its last iteration's value is shown).
fn ran_values(entry: &orla_ir::Function, machine: &eval::Machine<'_, '_>) -> Vec<bool> {
    fn walk(
        instructions: &[orla_ir::Instruction],
        live: bool,
        machine: &eval::Machine<'_, '_>,
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
                    let chosen = live && machine.entry_value(*condition)[0] != 0;
                    walk(&then.body, chosen, machine, ran);
                    walk(&otherwise.body, live && !chosen, machine, ran);
                }
                orla_ir::Op::Match {
                    scrutinee,
                    arms,
                    otherwise,
                } => {
                    let value = machine.entry_value(*scrutinee)[0] as i32;
                    let mut taken = false;
                    for (values, block) in arms {
                        let chosen = live && !taken && values.contains(&value);
                        taken |= chosen;
                        walk(&block.body, chosen, machine, ran);
                    }
                    walk(&otherwise.body, live && !taken, machine, ran);
                }
                orla_ir::Op::Loop { until, body, .. } => {
                    if let Some(until) = until {
                        walk(&until.body, live, machine, ran);
                    }
                    walk(&body.body, live, machine, ran);
                }
                _ => {}
            }
        }
    }
    let mut ran = vec![false; entry.slot_count()];
    walk(&entry.body, true, machine, &mut ran);
    ran
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
    check_invocation(module, invocation)?;
    let mode = module.manifest.mode;
    let domain = match mode {
        Mode::Map => invocation.domain,
        Mode::Reduce => scope.size(invocation.inputs[0])?,
        Mode::Scan => check_scan(invocation, scope)?,
        Mode::Scatter { base, .. } => {
            check_scatter(invocation, scope, base as usize)?;
            scope.size(invocation.inputs[0])?
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
        .map(|port| port.ty.schema.words())
        .sum();
    let stride = match mode {
        Mode::Scatter { .. } => {
            address_words(domain) + module.manifest.outputs[0].ty.schema.words()
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
            }
            machine.run_scan_step(step, &carried, &mut out)?;
            carried.copy_from_slice(&out);
        }
    } else {
        machine.run_entry_words(at, &mut out)?;
    }
    let entry = &module.entry;
    let ran = ran_values(entry, &machine);
    let bindings = entry
        .bindings
        .iter()
        .map(|(id, name)| TraceValue {
            name: name.clone(),
            ty: type_of(entry, *id).unwrap_or(Type::Float),
            words: machine.entry_value(*id),
            ran: ran[id.0 as usize],
        })
        .collect();
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
    let results = entry
        .results
        .iter()
        .zip(names)
        .enumerate()
        .map(|(index, (id, name))| TraceValue {
            name,
            ty: type_of(entry, *id).unwrap_or(Type::Float),
            words: machine.entry_result(index),
            ran: true,
        })
        .collect();
    Ok(Trace {
        at: if mode == Mode::Scan {
            [at[0], 0, 0]
        } else {
            at
        },
        bindings,
        results,
    })
}

/// Checks a scan's invocation: it runs over its first input's rows and
/// writes as many; the domain says so. Every backend applies this.
pub fn check_scan(invocation: &Invocation, scope: &Scope<'_>) -> Result<Domain, ExecutionError> {
    let domain = scope.size(invocation.inputs[0])?;
    if domain != invocation.domain {
        return Err(ExecutionError::Device(format!(
            "a scan writes one element per element of its first input ({domain}); its domain cannot be {}",
            invocation.domain
        )));
    }
    Ok(domain)
}

/// A scan: the entry once per element of the first input, in order, each
/// step reading the previous step's result and writing its own. Ordered
/// work runs on one thread; that is its nature, not a limitation of the
/// backend.
fn scan(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
) -> Result<Vec<Handle>, ExecutionError> {
    let domain = check_scan(invocation, scope)?;
    let count = domain.count();
    let program = eval::Program::new(module, invocation, scope, domain)?;
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
    let memories = machine.take_memories();
    drop(machine);
    drop(program);
    let port = &module.manifest.outputs[0];
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_all(output, &words) {
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
            .and_then(|handle| scope.write_all(handle, &words).map(|()| handle));
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

/// Checks a scatter's invocation: it writes the extent of the input it
/// scatters into; the domain says so. Every backend applies this.
pub fn check_scatter(
    invocation: &Invocation,
    scope: &Scope<'_>,
    base: usize,
) -> Result<Domain, ExecutionError> {
    let domain = scope.size(invocation.inputs[base])?;
    if domain != invocation.domain {
        return Err(ExecutionError::Device(format!(
            "a scatter writes the extent of the input it scatters into ({domain}); its domain cannot be {}",
            invocation.domain
        )));
    }
    Ok(domain)
}

/// Words an address into a domain takes in a scatter's pair.
pub fn address_words(domain: Domain) -> usize {
    domain.shape().dims() as usize
}

/// The linear index a scattered address names, or `None` when it falls
/// outside the target (NaN included): the write is dropped, never clamped.
pub fn scatter_index(domain: Domain, address: &[u32]) -> Option<usize> {
    if domain.shape() == Shape::Rows {
        let i = address[0] as i32;
        return (i >= 0 && (i as usize) < domain.count()).then_some(i as usize);
    }
    let axis = |word: u32, limit: u32| -> Option<u32> {
        let v = f32::from_bits(word).floor();
        (v >= 0.0 && v < limit as f32).then_some(v as u32)
    };
    let x = axis(address[0], domain.width())?;
    let y = axis(address[1], domain.height())?;
    let z = if domain.shape() == Shape::Volume {
        axis(address[2], domain.depth())?
    } else {
        0
    };
    Some(((z * domain.height() + y) * domain.width() + x) as usize)
}

/// The total order `min` and `max` use over floats, as a key that orders
/// like the float: every negative below every positive, negative zero
/// below positive zero, NaN beyond the infinities by its bits. Both
/// backends compare by it, so a parallel minimum on the GPU and a
/// sequential one here pick the same word.
pub fn float_order_key(word: u32) -> i32 {
    let bits = word as i32;
    if bits >= 0 { bits } else { bits ^ 0x7fff_ffff }
}

/// Combines a scattered value into the element already there, scalar by
/// scalar; the one rule every backend follows.
pub fn combine_words(
    combine: Combine,
    scalars: &[Scalar],
    element: &mut [u32],
    value: &[u32],
) -> Result<(), ArithmeticError> {
    for ((slot, incoming), scalar) in element.iter_mut().zip(value).zip(scalars) {
        let (a, b) = (*slot, *incoming);
        *slot = match (combine, scalar) {
            (Combine::Last, _) => b,
            (Combine::Sum, Scalar::Float) => (f32::from_bits(a) + f32::from_bits(b)).to_bits(),
            (Combine::Sum, _) => (a as i32)
                .checked_add(b as i32)
                .ok_or(ArithmeticError::Overflow)? as u32,
            (Combine::Min, Scalar::Float) => {
                if float_order_key(b) < float_order_key(a) {
                    b
                } else {
                    a
                }
            }
            (Combine::Min, _) => (a as i32).min(b as i32) as u32,
            (Combine::Max, Scalar::Float) => {
                if float_order_key(b) > float_order_key(a) {
                    b
                } else {
                    a
                }
            }
            (Combine::Max, _) => (a as i32).max(b as i32) as u32,
        };
    }
    Ok(())
}

/// A scatter: the entry over the first input, then every address and
/// value applied to a copy of the target, one after another in element
/// order. The order is the rule, so the result is the same on every
/// backend, floats included.
fn scatter(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    stride: usize,
    base: usize,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let target = check_scatter(invocation, scope, base)?;
    let sources = scope.size(invocation.inputs[0])?;
    let address = address_words(target);
    let program = eval::Program::new(module, invocation, scope, sources)?;
    let pairs = compute(&program, &[address + stride])?;
    drop(program);
    let mut words = scope.words(invocation.inputs[base])?.to_vec();
    let port = &module.manifest.outputs[0];
    let scalars = port.ty.schema.scalars();
    for (index, pair) in pairs.chunks_exact(address + stride).enumerate() {
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
    if let Err(error) = scope.write_all(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
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
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(rows)
        .min(element_count.div_ceil(PARALLEL_THRESHOLD))
        .max(1);
    let rows_per_thread = rows.div_ceil(threads);
    let run_rows = |first_row: usize, chunk: &mut [u32]| -> Result<(), ExecutionError> {
        let mut machine = eval::Machine::new(program);
        for (offset, row) in chunk.chunks_exact_mut(row_words.max(1)).enumerate() {
            let flat = (first_row + offset) as u32;
            let (y, z) = (flat % height, flat / height);
            for x in 0..domain.width() {
                let slot = x as usize * stride;
                machine.run_entry_words([x, y, z], &mut row[slot..slot + stride])?;
            }
        }
        Ok(())
    };
    if threads == 1 {
        run_rows(0, &mut results)?;
        return Ok(results);
    }
    let chunk_len = rows_per_thread * row_words.max(1);
    let outcomes: Vec<Result<(), ExecutionError>> = std::thread::scope(|s| {
        let handles: Vec<_> = results
            .chunks_mut(chunk_len)
            .enumerate()
            .map(|(index, chunk)| {
                let run_rows = &run_rows;
                s.spawn(move || run_rows(index * rows_per_thread, chunk))
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

/// Checks that an invocation matches the manifest: input and parameter
/// counts, parameter types, and a non-empty domain. Every backend applies
/// the same rule.
pub fn check_invocation(module: &Validated, invocation: &Invocation) -> Result<(), ExecutionError> {
    let manifest = &module.manifest;
    if invocation.inputs.len() != manifest.inputs.len() {
        return Err(ExecutionError::InputCount {
            expected: manifest.inputs.len(),
            actual: invocation.inputs.len(),
        });
    }
    if invocation.params.len() != manifest.params.len() {
        return Err(ExecutionError::ParamCount {
            expected: manifest.params.len(),
            actual: invocation.params.len(),
        });
    }
    for (param, value) in manifest.params.iter().zip(&invocation.params) {
        if value.ty() != param.default.ty() {
            return Err(ExecutionError::ParamType {
                name: param.name.clone(),
                expected: param.default.ty(),
                actual: value.ty(),
            });
        }
    }
    if invocation.domain.is_empty() {
        return Err(ExecutionError::EmptyDomain);
    }
    if manifest.mode == Mode::Reduce {
        if invocation.domain != Domain::Rows(1) {
            return Err(ExecutionError::Device(
                "a reduction produces one element; its domain is 1".into(),
            ));
        }
        return Ok(());
    }
    if let Some(output) = manifest.outputs.first()
        && output.ty.shape != invocation.domain.shape()
    {
        return Err(ExecutionError::DomainShape {
            expected: output.ty.shape,
            actual: invocation.domain.shape(),
        });
    }
    Ok(())
}

fn release_all(scope: &mut Scope<'_>, outputs: &[Handle]) {
    for output in outputs {
        // The output was created through this scope, so release cannot be
        // denied; a Host failure here would mean the handle is already gone.
        let _ = scope.release(*output);
    }
}
