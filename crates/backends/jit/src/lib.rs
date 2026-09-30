#![warn(missing_docs)]
//! JIT backend.
//!
//! Compiles a [`Validated`] module's functions with Cranelift when the
//! Kernel is prepared and runs them on as many threads as pay off. The
//! contract is the CPU interpreter's, bit for bit: word layout, integer
//! checks, reads through the scope's gate, and libm. Native operations
//! run on the wrapped CPU backend.
//!
//! When Orla is itself WebAssembly, the code is WebAssembly (`wasm`),
//! loaded into Orla's function table; a table index stands for a
//! function pointer.

#[cfg(not(target_arch = "wasm32"))]
mod codegen;
mod failure;
mod frame;
mod layout;
#[cfg(target_arch = "wasm32")]
mod wasm;

#[cfg(not(target_arch = "wasm32"))]
use std::any::Any;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
use cranelift_codegen::isa::TargetIsa;
#[cfg(not(target_arch = "wasm32"))]
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

/// What the compiled entry reads besides its arguments: inputs,
/// parameters as words, context, output domain, and the failure slot.
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
    /// How many pads are connected.
    pads: u32,
    window: [f32; 2],
    scale: f32,
    buttons: u32,
    keys: [u32; 8],
    width: u32,
    height: u32,
    depth: u32,
    /// Zero, or the first failure code (see `failure`).
    error: u32,
    /// Loop iterations the running element has run, when the module
    /// counts them.
    work: u32,
    /// The element's count at its last look; blocks up to it are
    /// already the run's.
    looked: u32,
    /// The run's count of blocks, which the element adds a block to at
    /// a time, and the most it may reach ([`orla_rules::Run`]).
    run_blocks: *const std::sync::atomic::AtomicU32,
    run_limit: u32,
    /// The partial blocks of this thread's finished elements, summed and
    /// held back from the run's total.
    left: u64,
    /// The failing element's `x`, written by a map's entry as it leaves
    /// its run; the caller holds the other coordinates.
    error_at: u32,
    /// Where a scatter's `emit` keeps its pairs, each after its element;
    /// null where nothing emits.
    emitted: *mut Vec<u32>,
    /// The run's hold against the Host's room, covering emitted pairs.
    held: *const orla_rules::Held,
}

/// A map entry's native signature: the context, a run's first element
/// (`x`, and the shared `y`, `z`), its length, and where its output
/// words go, element after element. The entry walks the run itself.
type Entry = unsafe extern "C" fn(*mut Ctx, u32, u32, u32, u32, *mut u32);
/// A scan entry's: the context, the step's index, the carried words,
/// and where the next carried words go.
type ScanEntry = unsafe extern "C" fn(*mut Ctx, u32, *const u32, *mut u32);
/// A scan's `init`: the context and where the first carried words go.
type Init = unsafe extern "C" fn(*mut Ctx, *mut u32);
/// A reduction's `merge`: the context, the two values' words, and where
/// the merged words go.
type Merge = unsafe extern "C" fn(*mut Ctx, *const u32, *const u32, *mut u32);

/// The JIT backend: a target ISA to compile for.
pub struct JitBackend {
    #[cfg(not(target_arch = "wasm32"))]
    isa: Arc<dyn TargetIsa>,
}

impl std::fmt::Debug for JitBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JitBackend({})", self.target())
    }
}

#[cfg(target_arch = "wasm32")]
impl JitBackend {
    /// Opens the backend where the host loads WebAssembly.
    pub fn new() -> Result<Self, String> {
        if cfg!(feature = "wasm-host") {
            Ok(Self {})
        } else {
            Err("the JIT makes code the host loads, and this build has no such host".into())
        }
    }

    /// The target the code is compiled for, for diagnostics.
    pub fn target(&self) -> String {
        "wasm32".into()
    }
}

#[cfg(not(target_arch = "wasm32"))]
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

/// A module's compiled code. Its function pointers are valid while the
/// JIT module lives, which is as long as the Kernel version.
#[cfg(not(target_arch = "wasm32"))]
struct Compiled {
    module: Option<cranelift_jit::JITModule>,
    entry: usize,
    /// A scan's `init`.
    init: Option<usize>,
    /// A reduction's `merge`.
    merge: Option<usize>,
}

// The module is read-only after `prepare`; its code is called from any
// thread.
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Send for Compiled {}
#[cfg(not(target_arch = "wasm32"))]
unsafe impl Sync for Compiled {}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(module) = self.module.take() {
            // The runtime drops prepared code after its last executing
            // thread lets go.
            unsafe { module.free_memory() };
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Prepared for Compiled {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl Backend for JitBackend {
    fn name(&self) -> &'static str {
        "jit"
    }

    fn declines(&self, module: &Validated) -> Option<String> {
        frame::refusal(module)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        if let Some(reason) = frame::refusal(module) {
            return Err(reason);
        }
        let code = codegen::compile(module, self.isa.clone())?;
        Ok(Box::new(Compiled {
            module: Some(code.module),
            entry: code.entry,
            init: code.init,
            merge: code.merge,
        }))
    }

    #[cfg(target_arch = "wasm32")]
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        if let Some(reason) = frame::refusal(module) {
            return Err(reason);
        }
        Ok(Box::new(wasm::compile(module)?))
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        #[cfg(target_arch = "wasm32")]
        use wasm::Compiled;
        let Some(compiled) = prepared.as_any().downcast_ref::<Compiled>() else {
            return Err(ExecutionError::Device(
                "kernel was prepared by a different backend".into(),
            ));
        };
        check_invocation(module, invocation, scope)?;
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

/// Read-only data the threads' contexts point into, and the context
/// they start from. The vectors live here to outlive the run and are
/// read through the context's pointers.
#[allow(dead_code)]
struct Shared {
    views: Vec<View>,
    params: Vec<u32>,
    /// The memories' working copies, by input index; views point here.
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
    /// The run's hold beyond its Fields against the Host's room; boxed
    /// for the same reason.
    held: Box<orla_rules::Held>,
}

// The views point into the scope's Fields, borrowed for the run, and the
// params are this run's; both are read-only.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

/// The inputs' views and their context for a run over `domain`. Named
/// memories get working copies the views point at, so steps read their
/// own writes and the inputs stay intact.
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
    let held = Box::new(orla_rules::Held::new(scope.room()));
    // The vectors' heap buffers stay put when the vectors move into `Shared`.
    let template = Ctx {
        views: views.as_ptr(),
        params: params.as_ptr(),
        time: invocation.context.time,
        rate: invocation.context.rate,
        pointer: input.pointer,
        wheel: input.wheel,
        sticks: input.sticks,
        pads: input.pads,
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
        held: &*held,
    };
    Ok(Shared {
        views,
        params,
        memories,
        schemas,
        template,
        budget: module.manifest.budget,
        run,
        held,
    })
}

/// Runs the entry over every element of `domain` and returns `stride`
/// words per element, in element order (rows of every layer in turn).
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

/// The same, with a scatter's emitted pairs as [`Emitted`] runs, in
/// order.
fn compute_emitting(
    entry: Entry,
    shared: &Shared,
    domain: Domain,
    stride: usize,
) -> Result<(Vec<u32>, Vec<Emitted>), ExecutionError> {
    let mut results = vec![0u32; domain.count() * stride];
    let emitted = compute_into(entry, shared, domain, stride, &mut results)?;
    Ok((results, emitted))
}

/// How many threads `items` pieces of work are worth: every core, but
/// no fewer than `PARALLEL_THRESHOLD` pieces a thread.
fn workers(items: usize) -> usize {
    orla_rules::cores()
        .min(items.div_ceil(PARALLEL_THRESHOLD))
        .max(1)
}

/// A run of elements' first element, the element after its last, and
/// the pairs its elements emitted, each after the element it came from.
type Emitted = (usize, usize, Vec<u32>);

/// The same, into words the caller owns, such as a new Field's.
fn compute_into(
    entry: Entry,
    shared: &Shared,
    domain: Domain,
    stride: usize,
    results: &mut [u32],
) -> Result<Vec<Emitted>, ExecutionError> {
    let width = domain.width() as usize;
    let height = domain.height();
    let rows = height as usize * domain.depth() as usize;
    let element_count = width * rows;
    // Split by elements: a `Rows` domain is one long row.
    let threads = orla_rules::cores()
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
        pads: t.pads,
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
        held: t.held,
    };
    // One call per run of elements sharing a row.
    let run_elements = |first: usize, chunk: &mut [u32]| -> Result<Emitted, ExecutionError> {
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
                return Err(failure::failure(
                    ctx.error,
                    ctx.error_at,
                    z * height + y,
                    shared,
                ));
            }
            hand_left(&mut ctx, shared, LEFT_HANDED)?;
            done += run;
        }
        hand_left(&mut ctx, shared, 1)?;
        Ok((first, first + elements, emitted))
    };
    if threads == 1 {
        return Ok(vec![run_elements(0, results)?]);
    }
    {
        let chunk_len = per_thread * stride.max(1);
        let outcomes: Vec<Result<Emitted, ExecutionError>> = std::thread::scope(|s| {
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
    // A single output is written straight into its Field. The views
    // made earlier stay valid because each Field's words are their own
    // allocation, which creating another Field leaves in place.
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
    orla_rules::outputs_fit_words(module, domain, scope)?;
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

/// A reduction: the entry over the first input's domain, then a
/// pairwise fold with `merge` in the CPU's fixed tree, so the result is
/// thread-count independent. One `Rows` element.
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
    let count = domain.count();
    if !shared
        .held
        .take(orla_rules::held_values(true, count, stride))
    {
        return Err(shared.held.over());
    }
    let mut level = compute(entry, &shared, domain, stride)?;
    let mut count = count;
    // The tree is fixed, level by level; a level's pairs are split among
    // workers, and the reported failure is the earliest pair's.
    while count > 1 {
        let pairs = count / 2;
        let mut next = vec![0u32; count.div_ceil(2) * stride];
        let per = pairs.div_ceil(workers(pairs)).max(1);
        let merge_run = |first: usize, outs: &mut [u32]| -> Result<(), (usize, ExecutionError)> {
            let mut ctx = Ctx { ..shared.template };
            for (k, out) in outs.chunks_exact_mut(stride).enumerate() {
                let pair = first + k;
                let a = level[2 * pair * stride..].as_ptr();
                let b = level[(2 * pair + 1) * stride..].as_ptr();
                // `merge` reads `stride` words at `a` and `b` and writes
                // `stride` words at `out`.
                ctx.work = 0;
                unsafe { merge(&mut ctx, a, b, out.as_mut_ptr()) };
                if ctx.error != 0 {
                    return Err((pair, failure::failure(ctx.error, 0, 0, &shared)));
                }
                keep_left(&mut ctx, &shared).map_err(|error| (pair, error))?;
            }
            hand_left(&mut ctx, &shared, 1).map_err(|error| (first, error))
        };
        let merged = &mut next[..pairs * stride];
        let outcomes: Vec<Result<(), (usize, ExecutionError)>> = if per >= pairs {
            vec![merge_run(0, merged)]
        } else {
            std::thread::scope(|s| {
                let handles: Vec<_> = merged
                    .chunks_mut(per * stride)
                    .enumerate()
                    .map(|(run, outs)| {
                        let merge_run = &merge_run;
                        s.spawn(move || merge_run(run * per, outs))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|handle| handle.join().expect("reduce worker panicked"))
                    .collect()
            })
        };
        if let Some((_, error)) = outcomes
            .into_iter()
            .filter_map(Result::err)
            .min_by_key(|(pair, _)| *pair)
        {
            return Err(error);
        }
        if count % 2 == 1 {
            next[pairs * stride..].copy_from_slice(&level[(count - 1) * stride..count * stride]);
        }
        level = next;
        count = count.div_ceil(2);
    }
    shared.run.outcome()?;
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_values(output, &level) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// A prefix: the entry over every element of the first input, then
/// each element folded with all before it, by the CPU's rule. The rule
/// is associative, so any grouping gives the reference's words.
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
    if !shared
        .held
        .take(orla_rules::held_values(false, domain.count(), stride))
    {
        return Err(shared.held.over());
    }
    let mut values = compute(entry, &shared, domain, stride)?;
    shared.run.outcome()?;
    drop(shared);
    let scalars = port.ty.schema.scalars();
    // One `Int` an element folds on the words directly, anything else by
    // the CPU's rule; both are the same arithmetic, overflow included.
    let int = |f: fn(i32, i32) -> Option<i32>| {
        move |running: &mut [u32], value: &[u32]| -> Result<(), ArithmeticError> {
            let folded = f(running[0] as i32, value[0] as i32).ok_or(ArithmeticError::Overflow)?;
            running[0] = folded as u32;
            Ok(())
        }
    };
    let folded = match (scalars, combine) {
        ([orla_abi::Scalar::Int], Combine::Sum) => {
            fold_prefix(&mut values, stride, int(i32::checked_add))
        }
        ([orla_abi::Scalar::Int], Combine::Min) => {
            fold_prefix(&mut values, stride, int(|a, b| Some(a.min(b))))
        }
        ([orla_abi::Scalar::Int], Combine::Max) => {
            fold_prefix(&mut values, stride, int(|a, b| Some(a.max(b))))
        }
        _ => fold_prefix(&mut values, stride, |running, value| {
            orla_rules::combine_words(combine, scalars, running, value)
        }),
    };
    folded.map_err(|(element, error)| ExecutionError::Arithmetic {
        x: element as u32,
        y: 0,
        error,
    })?;
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_values(output, &values) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// Folds each element of `values` (`stride` words each) with all
/// before it, in place, failing where an in-order walk first fails.
/// In parallel: each run's total, the totals folded into each run's
/// start, then each run walked from its start. A failure in the totals
/// hands the whole to the sequential walk, which names the true one.
fn fold_prefix<E: Send>(
    values: &mut [u32],
    stride: usize,
    combine: impl Fn(&mut [u32], &[u32]) -> Result<(), E> + Sync,
) -> Result<(), (usize, E)> {
    let count = values.len() / stride.max(1);
    let walk = |values: &mut [u32], start: Option<&[u32]>, first: usize| {
        let mut running = match start {
            Some(start) => start.to_vec(),
            None => values[..stride].to_vec(),
        };
        let from = usize::from(start.is_none());
        for (k, value) in values.chunks_exact_mut(stride).enumerate().skip(from) {
            combine(&mut running, value).map_err(|error| (first + k, error))?;
            value.copy_from_slice(&running);
        }
        Ok(())
    };
    let per = count.div_ceil(workers(count)).max(1);
    if count <= 1 || per >= count {
        return walk(values, None, 0);
    }
    // Each run's total, from its own first element.
    let totals: Vec<Result<Vec<u32>, E>> = std::thread::scope(|s| {
        let handles: Vec<_> = values
            .chunks(per * stride)
            .map(|run| {
                let combine = &combine;
                s.spawn(move || {
                    let mut total = run[..stride].to_vec();
                    for value in run.chunks_exact(stride).skip(1) {
                        combine(&mut total, value)?;
                    }
                    Ok(total)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("prefix worker panicked"))
            .collect()
    });
    // Where each run after the first starts: the totals before it.
    let mut starts: Vec<Vec<u32>> = Vec::with_capacity(totals.len());
    let mut running: Option<Vec<u32>> = None;
    for total in totals {
        let Ok(total) = total else {
            return walk(values, None, 0);
        };
        running = Some(match running {
            None => total,
            Some(mut sum) => {
                if combine(&mut sum, &total).is_err() {
                    return walk(values, None, 0);
                }
                sum
            }
        });
        starts.push(running.clone().expect("set above"));
    }
    let outcomes: Vec<Result<(), (usize, E)>> = std::thread::scope(|s| {
        let handles: Vec<_> = values
            .chunks_mut(per * stride)
            .enumerate()
            .map(|(run, slice)| {
                let walk = &walk;
                let start = run.checked_sub(1).map(|before| &starts[before][..]);
                s.spawn(move || walk(slice, start, run * per))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("prefix worker panicked"))
            .collect()
    });
    // With the walk's starts, every run folds the walk's words up to its
    // first failure; the earliest is the walk's.
    match outcomes
        .into_iter()
        .filter_map(Result::err)
        .min_by_key(|(element, _)| *element)
    {
        Some(failure) => Err(failure),
        None => Ok(()),
    }
}

/// A scatter: the entry over the first input into address-value pairs,
/// applied to a copy of the target in element order by the CPU's rule.
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
    // Held against the Host's room: the returned pairs (one an element)
    // before the run, emitted pairs as they come, and the parts' copies
    // of landing pairs before sorting.
    if !shared.held.take(sources.count() * pair_words) {
        return Err(shared.held.over());
    }
    let (pairs, runs) = compute_emitting(entry, &shared, sources, pair_words)?;
    shared.run.outcome()?;
    let mut words = scope.values(invocation.inputs[base])?.into_owned();
    let scalars = port.ty.schema.scalars();
    // The target is cut into one part a worker. Each run sorts its pairs
    // in element order (emitted before returned, as the interpreter does)
    // into the part each lands in. Each part takes its pairs run by run,
    // so every target element sees the sequential order and has one
    // writer.
    let count = target.count();
    let pair_count = pairs.len() / pair_words.max(1)
        + runs
            .iter()
            .map(|(_, _, emitted)| emitted.len() / (pair_words + 1))
            .sum::<usize>();
    let parts = workers(pair_count).min(count.max(1));
    let span = count.div_ceil(parts).max(1);
    // A record: where the pair lands, the element it came from, the value.
    let record = stride + 2;
    if !shared.held.take(pair_count * record) {
        return Err(shared.held.over());
    }
    // A sum's whole totals, one a target integer, held too.
    let summed = combine == Combine::Sum;
    if summed
        && !shared
            .held
            .take(orla_rules::Sums::held_words(scalars, count))
    {
        return Err(shared.held.over());
    }
    drop(shared);
    let sort = |(first, end, emitted): &Emitted| -> Vec<Vec<u32>> {
        let mut buckets = vec![Vec::new(); parts];
        let mut put = |pair: &[u32], index: usize| {
            if let Some(at) = orla_rules::scatter_index(target, &pair[..address]) {
                let bucket = &mut buckets[at / span];
                bucket.push(at as u32);
                bucket.push(index as u32);
                bucket.extend_from_slice(&pair[address..]);
            }
        };
        let mut kept = emitted.chunks_exact(pair_words + 1).peekable();
        for index in *first..*end {
            while let Some(emitted) = kept.next_if(|emitted| emitted[0] as usize == index) {
                put(&emitted[1..], index);
            }
            put(&pairs[index * pair_words..(index + 1) * pair_words], index);
        }
        buckets
    };
    let sorted: Vec<Vec<Vec<u32>>> = if runs.len() == 1 {
        vec![sort(&runs[0])]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = runs
                .iter()
                .map(|run| {
                    let sort = &sort;
                    s.spawn(move || sort(run))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("scatter worker panicked"))
                .collect()
        })
    };
    let apply = |part: usize, slice: &mut [u32]| -> Result<(), (usize, ArithmeticError)> {
        // A sum is each element's whole total; the failure is the first
        // target element whose total overflows.
        if summed {
            let mut sums = orla_rules::Sums::new(scalars, slice);
            for run in &sorted {
                for pair in run[part].chunks_exact(record) {
                    sums.add(slice, pair[0] as usize - part * span, &pair[2..]);
                }
            }
            return sums
                .finish(slice)
                .map_err(|(element, error)| (part * span + element, error));
        }
        for run in &sorted {
            for pair in run[part].chunks_exact(record) {
                let at = pair[0] as usize - part * span;
                orla_rules::combine_words(
                    combine,
                    scalars,
                    &mut slice[at * stride..(at + 1) * stride],
                    &pair[2..],
                )
                .map_err(|error| (pair[1] as usize, error))?;
            }
        }
        Ok(())
    };
    let outcomes: Vec<Result<(), (usize, ArithmeticError)>> = if parts == 1 {
        vec![apply(0, &mut words)]
    } else {
        std::thread::scope(|s| {
            let handles: Vec<_> = words
                .chunks_mut(span * stride)
                .enumerate()
                .map(|(part, slice)| {
                    let apply = &apply;
                    s.spawn(move || apply(part, slice))
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().expect("scatter worker panicked"))
                .collect()
        })
    };
    // The failure is the one the sequential order meets first: the
    // earliest source element's, or under a sum the target's.
    if let Some((index, error)) = outcomes
        .into_iter()
        .filter_map(Result::err)
        .min_by_key(|(index, _)| *index)
    {
        let width = if summed {
            target.width()
        } else {
            sources.width()
        } as usize;
        return Err(ExecutionError::Arithmetic {
            x: (index % width) as u32,
            y: (index / width) as u32,
            error,
        });
    }
    let output = scope.create(port.ty.schema.clone(), target)?;
    if let Err(error) = scope.write_values(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// Partial-block work a thread holds before handing it to the run, so
/// the shared total is touched rarely.
const LEFT_HANDED: u64 = 64 * orla_ir::WORK_BLOCK as u64;

/// Hands a context's held partial-block work to the run once it is
/// `at_least`; fails once the run is past its total.
fn hand_left(ctx: &mut Ctx, shared: &Shared, at_least: u64) -> Result<(), ExecutionError> {
    if ctx.left < at_least.max(1) {
        return Ok(());
    }
    match shared.run.settle(std::mem::take(&mut ctx.left)) {
        true => Err(shared.run.overrun()),
        false => Ok(()),
    }
}

/// Holds the partial block of a merge, step or `init`, as the map's code
/// does for its elements, and resets the work count.
fn keep_left(ctx: &mut Ctx, shared: &Shared) -> Result<(), ExecutionError> {
    ctx.left += u64::from(ctx.work % orla_ir::WORK_BLOCK);
    ctx.work = 0;
    ctx.looked = 0;
    hand_left(ctx, shared, LEFT_HANDED)
}

/// Runs a scan on one thread: `init` for the first carried words, then
/// the entry once per element of the first input in order, each step
/// reading the previous result. Returns the output and the memories, in
/// the manifest's order.
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
        Err(failure::failure(ctx.error, index, 0, &shared))
    };
    // `init` writes exactly `stride` words; each step reads `stride` words
    // of the carried value and writes `stride` words of the next.
    ctx.work = 0;
    unsafe { init(&mut ctx, carried.as_mut_ptr()) };
    failure(&ctx, 0)?;
    keep_left(&mut ctx, &shared)?;
    let mut words = vec![0u32; count * stride];
    for index in 0..count {
        let out = &mut words[index * stride..(index + 1) * stride];
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

// ---- Helpers the compiled code calls by symbol. Each resolves an
// input's address as the CPU interpreter does. ----

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

/// `at` on rows: the index clamped into range.
unsafe extern "C" fn orla_jit_read_rows(ctx: *const Ctx, input: u32, i: i32, out: *mut u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    let index = i.clamp(0, count as i32 - 1) as usize;
    unsafe { copy_out(view, index, out) };
}

/// `sample`: this element's coordinates, clamped into the input.
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
        let (lower, upper, weight) = orla_rules::linear_axis(v, limit);
        (lower as usize, upper as usize, weight)
    };
    let (x0, x1, fx) = axis(x, view.width);
    let (y0, y1, fy) = axis(y, view.height);
    let (z0, z1, fz) = axis(z, view.depth);
    let stride = view.stride as usize;
    let (width, height) = (view.width as usize, view.height as usize);
    // An element of up to 64 words blends on the stack, a wider one on
    // the heap.
    let (mut sums_here, mut sums_heap, mut element_here, mut element_heap);
    let (sums, element): (&mut [f32], &mut [u32]) = if stride <= 64 {
        sums_here = [0.0f32; 64];
        element_here = [0u32; 64];
        (&mut sums_here[..stride], &mut element_here[..stride])
    } else {
        sums_heap = vec![0.0f32; stride];
        element_heap = vec![0u32; stride];
        (&mut sums_heap, &mut element_heap)
    };
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
        // Words kept whole are read in place; narrow ones are widened first.
        let words: &[u32] = if view.schema.is_null() {
            unsafe { std::slice::from_raw_parts(view.words.add(index * stride), stride) }
        } else {
            unsafe { copy_out(view, index, element.as_mut_ptr()) };
            element
        };
        for (slot, word) in sums.iter_mut().zip(words) {
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
    if ctx.emitted.is_null() || ctx.error != 0 {
        return;
    }
    // Held with the word naming its element; past the room, the element
    // stops as a failure does.
    if !ctx.held.is_null() && !unsafe { &*ctx.held }.take(n as usize + 1) {
        ctx.error = failure::HELD_OVER;
        return;
    }
    let emitted = unsafe { &mut *ctx.emitted };
    emitted.push(element);
    emitted.extend_from_slice(unsafe { std::slice::from_raw_parts(pair, n as usize) });
}

/// `put` into a rows memory: the element's words replaced; an index
/// outside the memory is dropped.
unsafe extern "C" fn orla_jit_put_rows(ctx: *const Ctx, input: u32, i: i32, value: *const u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    if i < 0 || i as u32 >= count {
        return;
    }
    unsafe { copy_in(view, i as usize, value) };
}

/// `put` into a grid or volume memory: an address outside it (NaN
/// included) is dropped unclamped, as in the interpreter.
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

/// `fma` on floats and doubles: one rounding, as the interpreter's
/// `mul_add`. WebAssembly code calls this.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
extern "C" fn orla_jit_fmaf(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
extern "C" fn orla_jit_fma(a: f64, b: f64, c: f64) -> f64 {
    a.mul_add(b, c)
}

/// A half float's bits widened to a float's by the Host's rule.
extern "C" fn orla_jit_half(h: u32) -> u32 {
    orla_abi::encoding::f16_to_f32(h)
}

/// The helpers by symbol name, for the JIT to import.
pub(crate) fn symbols() -> Vec<(&'static str, *const u8)> {
    vec![
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
        ("orla_jit_fmaf", orla_jit_fmaf as *const u8),
        ("orla_jit_fma", orla_jit_fma as *const u8),
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
    pub const PADS: i32 = std::mem::offset_of!(Ctx, pads) as i32;
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
