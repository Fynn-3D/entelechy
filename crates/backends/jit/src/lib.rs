#![warn(missing_docs)]
//! JIT backend.
//!
//! Compiles a [`Validated`] module's functions to native code with
//! Cranelift once, when the Kernel is prepared, and runs the entry once
//! per output element from as many threads as the CPU backend uses. The
//! contract is the CPU interpreter's, bit for bit: the same word layout,
//! the same integer checks, the same reads through the scope's gate (the
//! inputs are borrowed once and handed to the code as pointers), and the
//! same libm. Only the map mode is compiled; a reduction, a scan, a
//! scatter and the native operations run on the CPU backend, which this
//! one wraps.

mod codegen;

use std::any::Any;
use std::sync::Arc;

use cranelift_codegen::isa::TargetIsa;
use cranelift_codegen::settings::{self, Configurable};
use orla_abi::{Domain, Handle, Invocation, Scope};
use orla_ir::{Combine, Mode, Validated};
use orla_rules::{ArithmeticError, ExecutionError, check_invocation};
use orla_rules::{Backend, NativeOp, Prepared};

/// Elements per thread below which the run stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// One input as the compiled code sees it: its memory and its extent.
#[repr(C)]
pub struct View {
    /// The memory as kept, packed where the Field keeps scalars narrow.
    words: *const u32,
    width: u32,
    height: u32,
    depth: u32,
    /// The Kernel's words per element.
    stride: u32,
    /// Bytes an element takes in memory: `stride * 4` when it is wide.
    bytes: u32,
    /// Words of memory behind `words`.
    len: usize,
    /// The layout, when it keeps scalars narrow; null when the memory
    /// holds the Kernel's words. What the helpers widen and narrow by.
    schema: *const orla_abi::Schema,
}

/// Everything the compiled entry reads besides its own arguments: the
/// inputs, the parameters as words, the context, the output domain, and
/// a slot for the first integer error.
#[repr(C)]
pub struct Ctx {
    views: *const View,
    params: *const u32,
    time: f32,
    rate: f32,
    pointer: [f32; 2],
    wheel: [f32; 2],
    /// A pad's axis pairs: the left stick, the right, the triggers.
    sticks: [[f32; 2]; 3],
    window: [f32; 2],
    scale: f32,
    buttons: u32,
    keys: [u32; 8],
    width: u32,
    height: u32,
    depth: u32,
    /// Zero, or the first failure: 1 division by zero, 2 overflow, 3
    /// more loop iterations than the budget, 4 the run past its total.
    error: u32,
    /// Loop iterations the running element has run, when the module
    /// counts them.
    work: u32,
    /// The element's count when it last looked: the blocks up to it are
    /// the run's already.
    looked: u32,
    /// The run's count of blocks, which the element adds a block to at
    /// a time, and the most it may reach ([`orla_rules::Run`]).
    run_blocks: *const std::sync::atomic::AtomicU32,
    run_limit: u32,
    /// What this thread's finished elements ran short of their blocks,
    /// summed, not yet handed to the run's total.
    left: u64,
    /// The `x` the failure was seen at, which a map's entry writes as it
    /// leaves its run; the caller knows the rest of the coordinates.
    error_at: u32,
    /// Where a scatter's `emit` keeps its pairs, each after the element
    /// it came from; null where nothing emits.
    emitted: *mut Vec<u32>,
}

/// A map entry's native signature: the context, where a run of elements
/// starts (`x`, and the `y` and `z` they share), how many there are,
/// and where their output words go, one element after another. The
/// entry walks the run itself: a million elements is one call, so the
/// reads it does not have to repeat are hoisted out of it.
type Entry = unsafe extern "C" fn(*mut Ctx, u32, u32, u32, u32, *mut u32);
/// A scan entry's: the context, the step's index, the carried words, and
/// where the next carried words go.
type ScanEntry = unsafe extern "C" fn(*mut Ctx, u32, *const u32, *mut u32);
/// A scan's `init`: the context and where the first carried words go.
type Init = unsafe extern "C" fn(*mut Ctx, *mut u32);
/// A reduction's `merge`: the context, the two values' words, and where
/// the merged words go.
type Merge = unsafe extern "C" fn(*mut Ctx, *const u32, *const u32, *mut u32);

/// The JIT backend: a target ISA to compile for.
pub struct JitBackend {
    isa: Arc<dyn TargetIsa>,
}

impl std::fmt::Debug for JitBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitBackend({})", self.isa.triple())
    }
}

impl JitBackend {
    /// Opens the backend for this machine.
    pub fn new() -> Result<Self, String> {
        let mut flags = settings::builder();
        flags
            .set("opt_level", "speed")
            .map_err(|error| error.to_string())?;
        let isa = cranelift_native::builder()
            .map_err(|error| error.to_string())?
            .finish(settings::Flags::new(flags))
            .map_err(|error| error.to_string())?;
        Ok(Self { isa })
    }

    /// The target the code is compiled for, for diagnostics.
    pub fn target(&self) -> String {
        self.isa.triple().to_string()
    }
}

/// A module's compiled code. The function pointer stays valid as long as
/// the JIT module lives, and it lives as long as the Kernel version.
struct Compiled {
    module: Option<cranelift_jit::JITModule>,
    entry: usize,
    /// A scan's `init`.
    init: Option<usize>,
    /// A reduction's `merge`.
    merge: Option<usize>,
}

// The module is only read after finalization, and the code it holds is
// called from any thread; nothing in it is mutated after `prepare`.
unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // No execution holds this version any more: the runtime drops
            // the prepared code after the last executing thread lets go.
            unsafe { module.free_memory() };
        }
    }
}

impl Prepared for Compiled {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Backend for JitBackend {
    fn name(&self) -> &'static str {
        "jit"
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let code = codegen::compile(module, self.isa.clone())?;
        Ok(Box::new(Compiled {
            module: Some(code.module),
            entry: code.entry,
            init: code.init,
            merge: code.merge,
        }))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        let Some(compiled) = prepared.as_any().downcast_ref::<Compiled>() else {
            return Err(ExecutionError::Device(
                "kernel was prepared by a different backend".into(),
            ));
        };
        check_invocation(module, invocation)?;
        match module.manifest.mode {
            Mode::Map => run(compiled.entry, module, invocation, scope),
            Mode::Scan => {
                let init = compiled.init.expect("a scan compiles its init");
                run_scan(compiled.entry, init, module, invocation, scope)
            }
            Mode::Reduce => {
                let merge = compiled.merge.expect("a reduction compiles its merge");
                run_reduce(compiled.entry, merge, module, invocation, scope)
            }
            Mode::Prefix { combine } => {
                run_prefix(compiled.entry, module, invocation, scope, combine)
            }
            Mode::Scatter { base, combine } => run_scatter(
                compiled.entry,
                module,
                invocation,
                scope,
                base as usize,
                combine,
            ),
        }
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

/// Words of a parameter, as the CPU lays them out.
fn param_words(value: &orla_abi::Value) -> Vec<u32> {
    value.words()
}

/// Shared read-only data the threads' contexts point into, and the
/// context they start from. The vectors are read through the pointers
/// in the context, never by name; they live here so they outlive the run.
#[allow(dead_code)]
struct Shared {
    views: Vec<View>,
    params: Vec<u32>,
    /// The memories' working copies, by input index; the views point into
    /// them.
    memories: Vec<(usize, Vec<u32>)>,
    /// The layouts of the inputs kept narrow; the views point at them.
    /// Boxed so a view's pointer stays good while the vector grows.
    #[allow(clippy::vec_box)]
    schemas: Vec<Box<orla_abi::Schema>>,
    template: Ctx,
    /// The loop iterations an element may run, for naming a failure.
    budget: u32,
    /// The run's total; boxed so the contexts' pointer to it stays good.
    run: Box<orla_rules::Run>,
}

// The views point into the scope's Fields, borrowed for the run, and the
// params are this run's; both are only read.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// The views of the inputs and the context they hang from, for a run
/// over `domain`. The memories named get working copies of their own,
/// which the views point at instead, so the steps' writes are seen by
/// their reads and the inputs stay untouched.
fn shared(
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    domain: Domain,
) -> Result<Shared, ExecutionError> {
    let mut views = Vec::with_capacity(invocation.inputs.len());
    let mut memories = Vec::new();
    let mut schemas = Vec::new();
    for (index, handle) in invocation.inputs.iter().enumerate() {
        let port = &module.manifest.inputs[index];
        let words = scope.storage(*handle)?;
        let len = words.len();
        let extent = scope.size(*handle)?;
        let layout = scope.schema(*handle)?;
        let stride = layout.value_words() as u32;
        let bytes = layout.element_bytes() as u32;
        let words = if module.manifest.memories.contains(&port.name) {
            let mut copy = words.to_vec();
            let pointer = copy.as_mut_ptr();
            memories.push((index, copy));
            pointer.cast_const()
        } else {
            words.as_ptr()
        };
        // A box does not move when the vector of them grows.
        let schema = if layout.is_wide() {
            std::ptr::null()
        } else {
            let boxed = Box::new(layout);
            let pointer: *const orla_abi::Schema = &*boxed;
            schemas.push(boxed);
            pointer
        };
        views.push(View {
            words,
            width: extent.width(),
            height: extent.height(),
            depth: extent.depth(),
            stride,
            bytes,
            len,
            schema,
        });
    }
    let params: Vec<u32> = invocation.params.iter().flat_map(param_words).collect();
    let input = &invocation.context.input;
    let run = Box::new(orla_rules::Run::new(scope.work()));
    // The vectors' heap buffers stay put when the vectors move into `Shared`.
    let template = Ctx {
        views: views.as_ptr(),
        params: params.as_ptr(),
        time: invocation.context.time,
        rate: invocation.context.rate,
        pointer: input.pointer,
        wheel: input.wheel,
        sticks: input.sticks,
        window: invocation.context.window,
        scale: invocation.context.scale,
        buttons: input.button_bits(),
        keys: input.key_words(),
        width: domain.width(),
        height: domain.height(),
        depth: domain.depth(),
        error: 0,
        work: 0,
        looked: 0,
        run_blocks: run.blocks(),
        run_limit: run.limit(),
        left: 0,
        error_at: 0,
        emitted: std::ptr::null_mut(),
    };
    Ok(Shared {
        views,
        params,
        memories,
        schemas,
        template,
        budget: module.manifest.budget,
        run,
    })
}

/// Runs the compiled entry over every element of `domain`, on as many
/// threads as pay off, and returns `stride` words per element, element
/// after element (rows of every layer one after another).
fn compute(
    entry: Entry,
    shared: &Shared,
    domain: Domain,
    stride: usize,
) -> Result<Vec<u32>, ExecutionError> {
    let mut results = vec![0u32; domain.count() * stride];
    compute_into(entry, shared, domain, stride, &mut results)?;
    Ok(results)
}

/// The same, with what a scatter's elements emitted: each run of elements
/// gives its pairs, each after the element it came from, and the runs
/// come in order.
fn compute_emitting(
    entry: Entry,
    shared: &Shared,
    domain: Domain,
    stride: usize,
) -> Result<(Vec<u32>, Vec<Vec<u32>>), ExecutionError> {
    let mut results = vec![0u32; domain.count() * stride];
    let emitted = compute_into(entry, shared, domain, stride, &mut results)?;
    Ok((results, emitted))
}

/// The same, into words the caller owns: a Field it just created, so a
/// map's output is written once instead of into a buffer that is then
/// copied into the Field.
fn compute_into(
    entry: Entry,
    shared: &Shared,
    domain: Domain,
    stride: usize,
    results: &mut [u32],
) -> Result<Vec<Vec<u32>>, ExecutionError> {
    let width = domain.width() as usize;
    let height = domain.height();
    let rows = height as usize * domain.depth() as usize;
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
    let fresh = |t: &Ctx| Ctx {
        views: t.views,
        params: t.params,
        time: t.time,
        rate: t.rate,
        pointer: t.pointer,
        wheel: t.wheel,
        sticks: t.sticks,
        window: t.window,
        scale: t.scale,
        buttons: t.buttons,
        keys: t.keys,
        width: t.width,
        height: t.height,
        depth: t.depth,
        error: 0,
        work: 0,
        looked: 0,
        run_blocks: t.run_blocks,
        run_limit: t.run_limit,
        left: 0,
        error_at: 0,
        emitted: std::ptr::null_mut(),
    };
    // One call a run of elements that share a row: a `Rows` domain is
    // one run of a million, a grid one run a row.
    let run_elements = |first: usize, chunk: &mut [u32]| -> Result<Vec<u32>, ExecutionError> {
        let mut ctx = fresh(&shared.template);
        let mut emitted: Vec<u32> = Vec::new();
        ctx.emitted = &mut emitted;
        let elements = chunk.len() / stride.max(1);
        let mut done = 0usize;
        while done < elements {
            let element = first + done;
            let x = (element % width.max(1)) as u32;
            let row = (element / width.max(1)) as u32;
            let (y, z) = (row % height, row / height);
            let run = (width.max(1) - x as usize).min(elements - done);
            // The entry writes exactly `stride` words an element at
            // `out`, element after element, reads the inputs through the
            // views, and writes nothing else.
            unsafe {
                entry(
                    &mut ctx,
                    x,
                    y,
                    z,
                    run as u32,
                    chunk[done * stride.max(1)..].as_mut_ptr(),
                )
            };
            if ctx.error != 0 {
                return Err(failure(ctx.error, ctx.error_at, z * height + y, shared));
            }
            hand_left(&mut ctx, shared, LEFT_HANDED)?;
            done += run;
        }
        hand_left(&mut ctx, shared, 1)?;
        Ok(emitted)
    };
    if threads == 1 {
        return Ok(vec![run_elements(0, results)?]);
    }
    {
        let chunk_len = per_thread * stride.max(1);
        let outcomes: Vec<Result<Vec<u32>, ExecutionError>> = std::thread::scope(|s| {
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
                .map(|handle| handle.join().expect("element worker panicked"))
                .collect()
        });
        outcomes.into_iter().collect()
    }
}

/// A map: the entry over the invocation's domain, its words split into
/// the outputs.
fn run(
    entry: usize,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let entry: Entry = unsafe { std::mem::transmute::<usize, Entry>(entry) };
    let domain = invocation.domain;
    let shared = shared(module, invocation, scope, domain)?;
    let strides: Vec<usize> = module
        .manifest
        .outputs
        .iter()
        .map(|port| port.ty.schema.value_words())
        .collect();
    let stride: usize = strides.iter().sum();
    let count = domain.count();
    // One output is the common case, and then the entry writes straight
    // into the Field: no buffer of our own and no copy into it. The
    // views were made before this Field was, and they hold pointers
    // into the inputs' words: what keeps them good is that a Field's
    // words are an allocation of their own, so making another Field
    // moves nothing. A Host that ever put every Field's words in one
    // place would have to give this its own reading of them.
    if let [port] = &module.manifest.outputs[..]
        && port.ty.schema.is_wide()
    {
        let handle = scope.create_for_fill(port.ty.schema.clone(), domain)?;
        let outcome = scope
            .storage_mut(handle)
            .map_err(ExecutionError::from)
            .and_then(|words| compute_into(entry, &shared, domain, stride, words))
            .and_then(|_| shared.run.outcome());
        return match outcome {
            Ok(()) => Ok(vec![handle]),
            Err(error) => {
                let _ = scope.release(handle);
                Err(error)
            }
        };
    }
    let results = compute(entry, &shared, domain, stride)?;
    shared.run.outcome()?;
    let mut outputs = Vec::with_capacity(module.manifest.outputs.len());
    let mut offset = 0;
    for (port, words) in module.manifest.outputs.iter().zip(&strides) {
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
                for output in outputs {
                    let _ = scope.release(output);
                }
                return Err(error.into());
            }
        }
    }
    Ok(outputs)
}

/// A reduction: the entry over the first input's domain, then a pairwise
/// fold with `merge` in the fixed tree the CPU uses, so the result does
/// not depend on how many threads ran the map. One `Rows` element.
fn run_reduce(
    entry: usize,
    merge: usize,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let entry: Entry = unsafe { std::mem::transmute::<usize, Entry>(entry) };
    let merge: Merge = unsafe { std::mem::transmute::<usize, Merge>(merge) };
    let domain = scope.size(invocation.inputs[0])?;
    let shared = shared(module, invocation, scope, domain)?;
    let port = &module.manifest.outputs[0];
    let stride = port.ty.schema.value_words();
    let mut level = compute(entry, &shared, domain, stride)?;
    let mut ctx = Ctx { ..shared.template };
    let mut count = domain.count();
    while count > 1 {
        let pairs = count / 2;
        let mut next = vec![0u32; count.div_ceil(2) * stride];
        for pair in 0..pairs {
            let a = level[2 * pair * stride..].as_ptr();
            let b = level[(2 * pair + 1) * stride..].as_ptr();
            let out = next[pair * stride..].as_mut_ptr();
            // `merge` reads `stride` words at `a` and `b` and writes
            // `stride` words at `out`.
            // A merge, a step and `init` each start their work anew.
            ctx.work = 0;
            unsafe { merge(&mut ctx, a, b, out) };
            if ctx.error != 0 {
                return Err(failure(ctx.error, 0, 0, &shared));
            }
            keep_left(&mut ctx, &shared)?;
        }
        if count % 2 == 1 {
            next[pairs * stride..].copy_from_slice(&level[(count - 1) * stride..count * stride]);
        }
        level = next;
        count = count.div_ceil(2);
    }
    hand_left(&mut ctx, &shared, 1)?;
    shared.run.outcome()?;
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_values(output, &level) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// A prefix: the entry over every element of the first input, then each
/// element folded with everything before it in element order, by the
/// rule the CPU applies. The rule is order-free, so walking it is one
/// grouping of many and gives the reference's words.
fn run_prefix(
    entry: usize,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let entry: Entry = unsafe { std::mem::transmute::<usize, Entry>(entry) };
    let domain = scope.size(invocation.inputs[0])?;
    let port = &module.manifest.outputs[0];
    let stride = port.ty.schema.value_words();
    let shared = shared(module, invocation, scope, domain)?;
    let mut values = compute(entry, &shared, domain, stride)?;
    shared.run.outcome()?;
    drop(shared);
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
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_values(output, &values) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// A scatter: the entry over the first input into pairs of address and
/// value, then every pair applied to a copy of the target, one after
/// another in element order, by the rule the CPU applies.
fn run_scatter(
    entry: usize,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    base: usize,
    combine: Combine,
) -> Result<Vec<Handle>, ExecutionError> {
    let entry: Entry = unsafe { std::mem::transmute::<usize, Entry>(entry) };
    let target = orla_rules::check_scatter(invocation, scope, base)?;
    let sources = invocation.domain;
    let address = orla_rules::address_words(target);
    let port = &module.manifest.outputs[0];
    let stride = port.ty.schema.value_words();
    let shared = shared(module, invocation, scope, sources)?;
    let pair_words = address + stride;
    let (pairs, emitted) = compute_emitting(entry, &shared, sources, pair_words)?;
    shared.run.outcome()?;
    drop(shared);
    let mut words = scope.values(invocation.inputs[base])?.into_owned();
    let scalars = port.ty.schema.scalars();
    let mut apply = |pair: &[u32], index: usize| -> Result<(), ExecutionError> {
        let Some(at) = orla_rules::scatter_index(target, &pair[..address]) else {
            return Ok(());
        };
        orla_rules::combine_words(
            combine,
            scalars,
            &mut words[at * stride..(at + 1) * stride],
            &pair[address..],
        )
        .map_err(|error| ExecutionError::Arithmetic {
            x: (index % sources.width() as usize) as u32,
            y: (index / sources.width() as usize) as u32,
            error,
        })
    };
    // Element by element: the pairs it emitted, each kept after its
    // index, then the one it returned.
    let mut kept = emitted
        .iter()
        .flat_map(|run| run.chunks_exact(pair_words + 1))
        .peekable();
    for (index, returned) in pairs.chunks_exact(pair_words).enumerate() {
        while let Some(record) = kept.next_if(|record| record[0] as usize == index) {
            apply(&record[1..], index)?;
        }
        apply(returned, index)?;
    }
    let output = scope.create(port.ty.schema.clone(), target)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// How much a thread keeps short of blocks before handing it to the run:
/// enough that the shared total is touched rarely.
const LEFT_HANDED: u64 = 64 * orla_ir::WORK_BLOCK as u64;

/// Hands the run what a context's elements ran short of their blocks,
/// once it is `at_least`: the run's failure once it has passed.
fn hand_left(ctx: &mut Ctx, shared: &Shared, at_least: u64) -> Result<(), ExecutionError> {
    if ctx.left < at_least.max(1) {
        return Ok(());
    }
    match shared.run.settle(std::mem::take(&mut ctx.left)) {
        true => Err(shared.run.overrun()),
        false => Ok(()),
    }
}

/// Keeps what a merge, a step or `init` ran short of a block, as the map's
/// code does for its elements, once its last look has handed the run its
/// blocks; then starts the next one's work anew.
fn keep_left(ctx: &mut Ctx, shared: &Shared) -> Result<(), ExecutionError> {
    ctx.left += u64::from(ctx.work % orla_ir::WORK_BLOCK);
    ctx.work = 0;
    ctx.looked = 0;
    hand_left(ctx, shared, LEFT_HANDED)
}

/// The failure a context's error code names.
/// What the code a run left in the context says, at the element named:
/// 1 a division by zero, 2 an overflow, 3 more loop iterations than the
/// budget, 4 the run past its total.
fn failure(code: u32, x: u32, y: u32, shared: &Shared) -> ExecutionError {
    let budget = shared.budget;
    match code {
        4 => shared.run.overrun(),
        1 => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::DivisionByZero,
        },
        3 => ExecutionError::Exhausted { x, y, budget },
        _ => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::Overflow,
        },
    }
}

/// Runs a scan: `init` for the first carried words, then the entry once
/// per element of the first input, in order, each step reading the
/// previous step's result and writing its own; then the output and the
/// memories as the steps left them, in the manifest's order. Ordered
/// work runs on one thread.
fn run_scan(
    entry: usize,
    init: usize,
    module: &Validated,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let entry: ScanEntry = unsafe { std::mem::transmute::<usize, ScanEntry>(entry) };
    let init: Init = unsafe { std::mem::transmute::<usize, Init>(init) };
    let domain = orla_rules::check_scan(invocation, scope)?;
    let shared = shared(module, invocation, scope, domain)?;
    let port = &module.manifest.outputs[0];
    let stride = port.ty.schema.value_words();
    let count = domain.count();
    let mut ctx = Ctx { ..shared.template };
    let mut carried = vec![0u32; stride];
    let failure = |ctx: &Ctx, index: u32| -> Result<(), ExecutionError> {
        if ctx.error == 0 {
            return Ok(());
        }
        Err(failure(ctx.error, index, 0, &shared))
    };
    // `init` writes exactly `stride` words; each step reads `stride` words
    // of the carried value and writes `stride` words of the next.
    // A merge, a step and `init` each start their work anew.
    ctx.work = 0;
    unsafe { init(&mut ctx, carried.as_mut_ptr()) };
    failure(&ctx, 0)?;
    keep_left(&mut ctx, &shared)?;
    let mut words = vec![0u32; count * stride];
    for index in 0..count {
        let out = &mut words[index * stride..(index + 1) * stride];
        // A merge, a step and `init` each start their work anew.
        ctx.work = 0;
        unsafe { entry(&mut ctx, index as u32, carried.as_ptr(), out.as_mut_ptr()) };
        failure(&ctx, index as u32)?;
        keep_left(&mut ctx, &shared)?;
        carried.copy_from_slice(out);
    }
    hand_left(&mut ctx, &shared, 1)?;
    shared.run.outcome()?;
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    let mut outputs = vec![output];
    for (input, words) in &shared.memories {
        let port = &module.manifest.inputs[*input];
        let created = scope.size(invocation.inputs[*input]).and_then(|extent| {
            scope
                .create(port.ty.schema.clone(), extent)
                .and_then(|handle| scope.write_storage(handle, words).map(|()| handle))
        });
        match created {
            Ok(handle) => outputs.push(handle),
            Err(error) => {
                for output in outputs {
                    let _ = scope.release(output);
                }
                return Err(error.into());
            }
        }
    }
    Ok(outputs)
}

// ---- Helpers the compiled code calls, by symbol. Each takes the context
// and an input index, resolves the address as the CPU interpreter does,
// and copies the element's words to `out`. ----

unsafe fn view<'a>(ctx: *const Ctx, input: u32) -> &'a View {
    unsafe { &*(*ctx).views.add(input as usize) }
}

/// Element `index` of a view as the Kernel's words, widened when it is
/// kept narrow.
unsafe fn copy_out(view: &View, index: usize, out: *mut u32) {
    let stride = view.stride as usize;
    if view.schema.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(view.words.add(index * stride), out, stride) };
        return;
    }
    let schema = unsafe { &*view.schema };
    let storage = unsafe { std::slice::from_raw_parts(view.words, view.len) };
    for k in 0..stride {
        unsafe { *out.add(k) = schema.load(storage, index, k) };
    }
}

/// The Kernel's words written as element `index` of a view's memory,
/// narrowed when it is kept narrow.
unsafe fn copy_in(view: &View, index: usize, value: *const u32) {
    let stride = view.stride as usize;
    let words = view.words.cast_mut();
    if view.schema.is_null() {
        unsafe { std::ptr::copy_nonoverlapping(value, words.add(index * stride), stride) };
        return;
    }
    let schema = unsafe { &*view.schema };
    let storage = unsafe { std::slice::from_raw_parts_mut(words, view.len) };
    for k in 0..stride {
        schema.store(storage, index, k, unsafe { *value.add(k) });
    }
}

fn clamp_axis(value: f32, limit: u32) -> u32 {
    if value.is_nan() {
        0
    } else {
        value.floor().clamp(0.0, (limit - 1) as f32) as u32
    }
}

/// `at` on a grid or volume: floored, clamped, NaN as zero.
unsafe extern "C" fn orla_jit_read(
    ctx: *const Ctx,
    input: u32,
    x: f32,
    y: f32,
    z: f32,
    out: *mut u32,
) {
    let view = unsafe { view(ctx, input) };
    let (cx, cy, cz) = (
        clamp_axis(x, view.width),
        clamp_axis(y, view.height),
        clamp_axis(z, view.depth),
    );
    let index = ((cz * view.height + cy) * view.width + cx) as usize;
    unsafe { copy_out(view, index, out) };
}

/// `at` on rows: the index clamped into range.
unsafe extern "C" fn orla_jit_read_rows(ctx: *const Ctx, input: u32, i: i32, out: *mut u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    let index = i.clamp(0, count as i32 - 1) as usize;
    unsafe { copy_out(view, index, out) };
}

/// `sample`: this element's own coordinates, clamped into the input.
unsafe extern "C" fn orla_jit_sample(
    ctx: *const Ctx,
    input: u32,
    x: u32,
    y: u32,
    z: u32,
    out: *mut u32,
) {
    let view = unsafe { view(ctx, input) };
    let (cx, cy, cz) = (
        x.min(view.width - 1),
        y.min(view.height - 1),
        z.min(view.depth - 1),
    );
    let index = ((cz * view.height + cy) * view.width + cx) as usize;
    unsafe { copy_out(view, index, out) };
}

/// `at_linear`: the CPU interpreter's blend of the neighbouring elements.
unsafe extern "C" fn orla_jit_linear(
    ctx: *const Ctx,
    input: u32,
    x: f32,
    y: f32,
    z: f32,
    out: *mut u32,
) {
    let view = unsafe { view(ctx, input) };
    let axis = |v: f32, limit: u32| -> (usize, usize, f32) {
        let t = if v.is_nan() { 0.0 } else { v - 0.5 };
        let lower = t.floor().clamp(0.0, (limit - 1) as f32);
        let upper = (lower + 1.0).min((limit - 1) as f32);
        let weight = (t - t.floor()).clamp(0.0, 1.0);
        let weight = if t < 0.0 { 0.0 } else { weight };
        (lower as usize, upper as usize, weight)
    };
    let (x0, x1, fx) = axis(x, view.width);
    let (y0, y1, fy) = axis(y, view.height);
    let (z0, z1, fz) = axis(z, view.depth);
    let stride = view.stride as usize;
    let (width, height) = (view.width as usize, view.height as usize);
    let mut sums = [0.0f32; 64];
    let sums = &mut sums[..stride];
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
        let mut element = [0u32; 64];
        unsafe { copy_out(view, (cz * height + cy) * width + cx, element.as_mut_ptr()) };
        for (slot, word) in sums.iter_mut().zip(&element) {
            *slot += f32::from_bits(*word) * weight;
        }
    }
    for (k, v) in sums.iter().enumerate() {
        unsafe { *out.add(k) = v.to_bits() };
    }
}

/// Whether a key is held, by code.
unsafe extern "C" fn orla_jit_key(ctx: *const Ctx, code: i32) -> i32 {
    if !(0..256).contains(&code) {
        return 0;
    }
    let keys = unsafe { &(*ctx).keys };
    ((keys[(code >> 5) as usize] >> (code & 31)) & 1) as i32
}

/// Whether a button is held: 0 left, 1 right, 2 middle of a pointer,
/// then a pad's from 8.
unsafe extern "C" fn orla_jit_button(ctx: *const Ctx, code: i32) -> i32 {
    if !(0..orla_abi::BUTTONS).contains(&code) {
        return 0;
    }
    let bits = unsafe { (*ctx).buttons };
    ((bits >> code) & 1) as i32
}

/// `emit`: the element's index and the pair's `n` words, kept in the
/// order they come.
unsafe extern "C" fn orla_jit_emit(ctx: *mut Ctx, element: u32, pair: *const u32, n: u32) {
    let ctx = unsafe { &mut *ctx };
    if ctx.emitted.is_null() {
        return;
    }
    let emitted = unsafe { &mut *ctx.emitted };
    emitted.push(element);
    emitted.extend_from_slice(unsafe { std::slice::from_raw_parts(pair, n as usize) });
}

/// `put` into a rows memory: the element's words replaced, an index
/// outside the memory dropped.
unsafe extern "C" fn orla_jit_put_rows(ctx: *const Ctx, input: u32, i: i32, value: *const u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    if i < 0 || i as u32 >= count {
        return;
    }
    unsafe { copy_in(view, i as usize, value) };
}

/// `put` into a grid or volume memory: an address outside it (NaN
/// included) dropped, never clamped, as the interpreter has it.
unsafe extern "C" fn orla_jit_put(
    ctx: *const Ctx,
    input: u32,
    x: f32,
    y: f32,
    z: f32,
    value: *const u32,
) {
    let view = unsafe { view(ctx, input) };
    let axis =
        |v: f32, limit: u32| -> Option<u32> { (v >= 0.0 && v < limit as f32).then_some(v as u32) };
    let (Some(cx), Some(cy), Some(cz)) = (
        axis(x, view.width),
        axis(y, view.height),
        axis(z, view.depth),
    ) else {
        return;
    };
    let index = ((cz * view.height + cy) * view.width + cx) as usize;
    unsafe { copy_in(view, index, value) };
}

/// The float error flag: the first failure wins.
unsafe extern "C" fn orla_jit_fail(ctx: *mut Ctx, code: u32) {
    unsafe {
        if (*ctx).error == 0 {
            (*ctx).error = code;
        }
    }
}

macro_rules! unary_math {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            /// Rust's `f32` math, so the JIT rounds as the interpreter does.
            extern "C" fn $name(x: f32) -> f32 {
                x.$method()
            }
        )*
    };
}

unary_math! {
    orla_jit_sinf => sin,
    orla_jit_cosf => cos,
    orla_jit_tanf => tan,
    orla_jit_asinf => asin,
    orla_jit_acosf => acos,
    orla_jit_atanf => atan,
    orla_jit_expf => exp,
    orla_jit_logf => ln,
    orla_jit_exp2f => exp2,
    orla_jit_log2f => log2,
    orla_jit_sinhf => sinh,
    orla_jit_coshf => cosh,
    orla_jit_tanhf => tanh,
}

macro_rules! unary_math64 {
    ($($name:ident => $method:ident),* $(,)?) => {
        $(
            /// Rust's `f64` math, so the JIT rounds as the interpreter does.
            extern "C" fn $name(x: f64) -> f64 {
                x.$method()
            }
        )*
    };
}

unary_math64! {
    orla_jit_sin => sin,
    orla_jit_cos => cos,
    orla_jit_tan => tan,
    orla_jit_asin => asin,
    orla_jit_acos => acos,
    orla_jit_atan => atan,
    orla_jit_exp => exp,
    orla_jit_log => ln,
    orla_jit_exp2 => exp2,
    orla_jit_log2 => log2,
    orla_jit_sinh => sinh,
    orla_jit_cosh => cosh,
    orla_jit_tanh => tanh,
}

/// `pow` on doubles.
extern "C" fn orla_jit_pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// `atan2` on doubles.
extern "C" fn orla_jit_atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// `%` on doubles: the remainder, truncated.
extern "C" fn orla_jit_fmod(a: f64, b: f64) -> f64 {
    a % b
}

/// `pow` on floats.
extern "C" fn orla_jit_powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

/// `atan2` on floats.
extern "C" fn orla_jit_atan2f(y: f32, x: f32) -> f32 {
    y.atan2(x)
}

/// `%` on floats: the remainder, truncated.
extern "C" fn orla_jit_fmodf(a: f32, b: f32) -> f32 {
    a % b
}

/// A half float's bits widened to a float's: the Host's own rule, so the
/// compiled code reads a half as the interpreter does.
extern "C" fn orla_jit_half(h: u32) -> u32 {
    orla_abi::encoding::f16_to_f32(h)
}

/// The helpers by symbol name, for the JIT to import.
pub(crate) fn symbols() -> Vec<(&'static str, *const u8)> {
    vec![
        ("orla_jit_read", orla_jit_read as *const u8),
        ("orla_jit_read_rows", orla_jit_read_rows as *const u8),
        ("orla_jit_sample", orla_jit_sample as *const u8),
        ("orla_jit_linear", orla_jit_linear as *const u8),
        ("orla_jit_key", orla_jit_key as *const u8),
        ("orla_jit_button", orla_jit_button as *const u8),
        ("orla_jit_put", orla_jit_put as *const u8),
        ("orla_jit_put_rows", orla_jit_put_rows as *const u8),
        ("orla_jit_emit", orla_jit_emit as *const u8),
        ("orla_jit_fail", orla_jit_fail as *const u8),
        ("orla_jit_sinf", orla_jit_sinf as *const u8),
        ("orla_jit_cosf", orla_jit_cosf as *const u8),
        ("orla_jit_tanf", orla_jit_tanf as *const u8),
        ("orla_jit_asinf", orla_jit_asinf as *const u8),
        ("orla_jit_acosf", orla_jit_acosf as *const u8),
        ("orla_jit_atanf", orla_jit_atanf as *const u8),
        ("orla_jit_expf", orla_jit_expf as *const u8),
        ("orla_jit_logf", orla_jit_logf as *const u8),
        ("orla_jit_exp2f", orla_jit_exp2f as *const u8),
        ("orla_jit_log2f", orla_jit_log2f as *const u8),
        ("orla_jit_sinhf", orla_jit_sinhf as *const u8),
        ("orla_jit_coshf", orla_jit_coshf as *const u8),
        ("orla_jit_tanhf", orla_jit_tanhf as *const u8),
        ("orla_jit_powf", orla_jit_powf as *const u8),
        ("orla_jit_atan2f", orla_jit_atan2f as *const u8),
        ("orla_jit_fmodf", orla_jit_fmodf as *const u8),
        ("orla_jit_half", orla_jit_half as *const u8),
        ("orla_jit_sin", orla_jit_sin as *const u8),
        ("orla_jit_cos", orla_jit_cos as *const u8),
        ("orla_jit_tan", orla_jit_tan as *const u8),
        ("orla_jit_asin", orla_jit_asin as *const u8),
        ("orla_jit_acos", orla_jit_acos as *const u8),
        ("orla_jit_atan", orla_jit_atan as *const u8),
        ("orla_jit_exp", orla_jit_exp as *const u8),
        ("orla_jit_log", orla_jit_log as *const u8),
        ("orla_jit_exp2", orla_jit_exp2 as *const u8),
        ("orla_jit_log2", orla_jit_log2 as *const u8),
        ("orla_jit_sinh", orla_jit_sinh as *const u8),
        ("orla_jit_cosh", orla_jit_cosh as *const u8),
        ("orla_jit_tanh", orla_jit_tanh as *const u8),
        ("orla_jit_pow", orla_jit_pow as *const u8),
        ("orla_jit_atan2", orla_jit_atan2 as *const u8),
        ("orla_jit_fmod", orla_jit_fmod as *const u8),
    ]
}

/// Byte offsets of the context's fields, for the compiled code's loads.
pub(crate) mod offsets {
    use super::{Ctx, View};
    pub const VIEWS: i32 = std::mem::offset_of!(Ctx, views) as i32;
    pub const PARAMS: i32 = std::mem::offset_of!(Ctx, params) as i32;
    pub const TIME: i32 = std::mem::offset_of!(Ctx, time) as i32;
    pub const RATE: i32 = std::mem::offset_of!(Ctx, rate) as i32;
    pub const POINTER: i32 = std::mem::offset_of!(Ctx, pointer) as i32;
    pub const WHEEL: i32 = std::mem::offset_of!(Ctx, wheel) as i32;
    pub const STICKS: i32 = std::mem::offset_of!(Ctx, sticks) as i32;
    pub const WINDOW: i32 = std::mem::offset_of!(Ctx, window) as i32;
    pub const SCALE: i32 = std::mem::offset_of!(Ctx, scale) as i32;
    pub const WIDTH: i32 = std::mem::offset_of!(Ctx, width) as i32;
    pub const HEIGHT: i32 = std::mem::offset_of!(Ctx, height) as i32;
    pub const DEPTH: i32 = std::mem::offset_of!(Ctx, depth) as i32;
    pub const ERROR: i32 = std::mem::offset_of!(Ctx, error) as i32;
    pub const ERROR_AT: i32 = std::mem::offset_of!(Ctx, error_at) as i32;
    pub const WORK: i32 = std::mem::offset_of!(Ctx, work) as i32;
    pub const LOOKED: i32 = std::mem::offset_of!(Ctx, looked) as i32;
    pub const RUN_BLOCKS: i32 = std::mem::offset_of!(Ctx, run_blocks) as i32;
    pub const RUN_LIMIT: i32 = std::mem::offset_of!(Ctx, run_limit) as i32;
    pub const LEFT: i32 = std::mem::offset_of!(Ctx, left) as i32;
    pub const VIEW_SIZE: i64 = std::mem::size_of::<View>() as i64;
    pub const VIEW_WORDS: i32 = std::mem::offset_of!(View, words) as i32;
    pub const VIEW_BYTES: i32 = std::mem::offset_of!(View, bytes) as i32;
    pub const VIEW_WIDTH: i32 = std::mem::offset_of!(View, width) as i32;
    pub const VIEW_HEIGHT: i32 = std::mem::offset_of!(View, height) as i32;
    pub const VIEW_DEPTH: i32 = std::mem::offset_of!(View, depth) as i32;
}
