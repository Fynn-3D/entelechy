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
use orla_runtime::{Backend, NativeOp, Prepared};

/// Elements per thread below which the run stays single-threaded.
const PARALLEL_THRESHOLD: usize = 4096;

/// One input as the compiled code sees it: its words and its extent.
#[repr(C)]
pub struct View {
    words: *const u32,
    width: u32,
    height: u32,
    depth: u32,
    stride: u32,
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
    window: [f32; 2],
    scale: f32,
    buttons: u32,
    keys: [u32; 8],
    width: u32,
    height: u32,
    depth: u32,
    /// Zero, or the first integer failure: 1 division by zero, 2 overflow.
    error: u32,
}

/// A map entry's native signature: the context, the element's
/// coordinates, and where its output words go.
type Entry = unsafe extern "C" fn(*mut Ctx, u32, u32, u32, *mut u32);
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

    fn as_any(&self) -> &dyn Any {
        self
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
            NativeOp::Rasterize => orla_cpu::raster::rasterize(invocation, scope),
        }
    }
}

/// Words of a parameter, as the CPU lays them out.
fn param_words(value: &orla_abi::Value) -> Vec<u32> {
    use orla_abi::Value;
    match value {
        Value::Float(v) => vec![v.to_bits()],
        Value::Int(v) => vec![*v as u32],
        Value::Bool(v) => vec![*v as u32],
        Value::Vec2(v) => v.iter().map(|c| c.to_bits()).collect(),
        Value::Vec3(v) => v.iter().map(|c| c.to_bits()).collect(),
        Value::Vec4(v) => v.iter().map(|c| c.to_bits()).collect(),
    }
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
    template: Ctx,
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
    for (index, handle) in invocation.inputs.iter().enumerate() {
        let port = &module.manifest.inputs[index];
        let words = scope.words(*handle)?;
        let extent = scope.size(*handle)?;
        let stride = scope.schema(*handle)?.words() as u32;
        let words = if module.manifest.memories.contains(&port.name) {
            let mut copy = words.to_vec();
            let pointer = copy.as_mut_ptr();
            memories.push((index, copy));
            pointer.cast_const()
        } else {
            words.as_ptr()
        };
        views.push(View {
            words,
            width: extent.width(),
            height: extent.height(),
            depth: extent.depth(),
            stride,
        });
    }
    let params: Vec<u32> = invocation.params.iter().flat_map(param_words).collect();
    let input = &invocation.context.input;
    // The vectors' heap buffers stay put when the vectors move into `Shared`.
    let template = Ctx {
        views: views.as_ptr(),
        params: params.as_ptr(),
        time: invocation.context.time,
        rate: invocation.context.rate,
        pointer: input.pointer,
        wheel: input.wheel,
        window: invocation.context.window,
        scale: invocation.context.scale,
        buttons: input.button_bits(),
        keys: input.key_words(),
        width: domain.width(),
        height: domain.height(),
        depth: domain.depth(),
        error: 0,
    };
    Ok(Shared {
        views,
        params,
        memories,
        template,
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
    let width = domain.width() as usize;
    let height = domain.height();
    let rows = height as usize * domain.depth() as usize;
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
    let fresh = |t: &Ctx| Ctx {
        views: t.views,
        params: t.params,
        time: t.time,
        rate: t.rate,
        pointer: t.pointer,
        wheel: t.wheel,
        window: t.window,
        scale: t.scale,
        buttons: t.buttons,
        keys: t.keys,
        width: t.width,
        height: t.height,
        depth: t.depth,
        error: 0,
    };
    let run_elements = |first: usize, chunk: &mut [u32]| -> Result<(), ExecutionError> {
        let mut ctx = fresh(&shared.template);
        for (offset, slot) in chunk.chunks_exact_mut(stride.max(1)).enumerate() {
            let element = first + offset;
            let x = (element % width.max(1)) as u32;
            let row = (element / width.max(1)) as u32;
            let (y, z) = (row % height, row / height);
            // The entry writes exactly `stride` words at `out`, reads the
            // inputs through the views, and writes nothing else.
            unsafe { entry(&mut ctx, x, y, z, slot.as_mut_ptr()) };
            if ctx.error != 0 {
                let error = if ctx.error == 1 {
                    ArithmeticError::DivisionByZero
                } else {
                    ArithmeticError::Overflow
                };
                return Err(ExecutionError::Arithmetic {
                    x,
                    y: z * height + y,
                    error,
                });
            }
        }
        Ok(())
    };
    if threads == 1 {
        run_elements(0, &mut results)?;
    } else {
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
                .map(|handle| handle.join().expect("element worker panicked"))
                .collect()
        });
        for outcome in outcomes {
            outcome?;
        }
    }
    Ok(results)
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
        .map(|port| port.ty.schema.words())
        .collect();
    let stride: usize = strides.iter().sum();
    let results = compute(entry, &shared, domain, stride)?;
    let count = domain.count();
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
            .and_then(|handle| scope.write_all(handle, &own).map(|()| handle));
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
    let stride = port.ty.schema.words();
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
            unsafe { merge(&mut ctx, a, b, out) };
            if ctx.error != 0 {
                return Err(ExecutionError::Arithmetic {
                    x: 0,
                    y: 0,
                    error: arithmetic(ctx.error),
                });
            }
        }
        if count % 2 == 1 {
            next[pairs * stride..].copy_from_slice(&level[(count - 1) * stride..count * stride]);
        }
        level = next;
        count = count.div_ceil(2);
    }
    let output = scope.create(port.ty.schema.clone(), Domain::Rows(1))?;
    if let Err(error) = scope.write_all(output, &level) {
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
    let stride = port.ty.schema.words();
    let shared = shared(module, invocation, scope, sources)?;
    let pairs = compute(entry, &shared, sources, address + stride)?;
    drop(shared);
    let mut words = scope.words(invocation.inputs[base])?.to_vec();
    let scalars = port.ty.schema.scalars();
    for (index, pair) in pairs.chunks_exact(address + stride).enumerate() {
        let Some(at) = orla_rules::scatter_index(target, &pair[..address]) else {
            continue;
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
        })?;
    }
    let output = scope.create(port.ty.schema.clone(), target)?;
    if let Err(error) = scope.write_all(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    Ok(vec![output])
}

/// The failure a context's error code names.
fn arithmetic(code: u32) -> ArithmeticError {
    if code == 1 {
        ArithmeticError::DivisionByZero
    } else {
        ArithmeticError::Overflow
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
    let stride = port.ty.schema.words();
    let count = domain.count();
    let mut ctx = Ctx { ..shared.template };
    let mut carried = vec![0u32; stride];
    let failure = |ctx: &Ctx, index: u32| -> Result<(), ExecutionError> {
        if ctx.error == 0 {
            return Ok(());
        }
        let error = if ctx.error == 1 {
            ArithmeticError::DivisionByZero
        } else {
            ArithmeticError::Overflow
        };
        Err(ExecutionError::Arithmetic {
            x: index,
            y: 0,
            error,
        })
    };
    // `init` writes exactly `stride` words; each step reads `stride` words
    // of the carried value and writes `stride` words of the next.
    unsafe { init(&mut ctx, carried.as_mut_ptr()) };
    failure(&ctx, 0)?;
    let mut words = vec![0u32; count * stride];
    for index in 0..count {
        let out = &mut words[index * stride..(index + 1) * stride];
        unsafe { entry(&mut ctx, index as u32, carried.as_ptr(), out.as_mut_ptr()) };
        failure(&ctx, index as u32)?;
        carried.copy_from_slice(out);
    }
    let output = scope.create(port.ty.schema.clone(), domain)?;
    if let Err(error) = scope.write_all(output, &words) {
        let _ = scope.release(output);
        return Err(error.into());
    }
    let mut outputs = vec![output];
    for (input, words) in &shared.memories {
        let port = &module.manifest.inputs[*input];
        let created = scope.size(invocation.inputs[*input]).and_then(|extent| {
            scope
                .create(port.ty.schema.clone(), extent)
                .and_then(|handle| scope.write_all(handle, words).map(|()| handle))
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
    let stride = view.stride as usize;
    unsafe { std::ptr::copy_nonoverlapping(view.words.add(index * stride), out, stride) };
}

/// `at` on rows: the index clamped into range.
unsafe extern "C" fn orla_jit_read_rows(ctx: *const Ctx, input: u32, i: i32, out: *mut u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    let index = i.clamp(0, count as i32 - 1) as usize;
    let stride = view.stride as usize;
    unsafe { std::ptr::copy_nonoverlapping(view.words.add(index * stride), out, stride) };
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
    let stride = view.stride as usize;
    unsafe { std::ptr::copy_nonoverlapping(view.words.add(index * stride), out, stride) };
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
        let start = ((cz * height + cy) * width + cx) * stride;
        for (k, slot) in sums.iter_mut().enumerate() {
            let word = unsafe { *view.words.add(start + k) };
            *slot += f32::from_bits(word) * weight;
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

/// Whether a pointer button is held: 0 left, 1 right, 2 middle.
unsafe extern "C" fn orla_jit_button(ctx: *const Ctx, code: i32) -> i32 {
    if !(0..3).contains(&code) {
        return 0;
    }
    let bits = unsafe { (*ctx).buttons };
    ((bits >> code) & 1) as i32
}

/// `put` into a rows memory: the element's words replaced, an index
/// outside the memory dropped.
unsafe extern "C" fn orla_jit_put_rows(ctx: *const Ctx, input: u32, i: i32, value: *const u32) {
    let view = unsafe { view(ctx, input) };
    let count = view.width * view.height * view.depth;
    if i < 0 || i as u32 >= count {
        return;
    }
    let stride = view.stride as usize;
    let words = view.words.cast_mut();
    unsafe { std::ptr::copy_nonoverlapping(value, words.add(i as usize * stride), stride) };
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
    let stride = view.stride as usize;
    let words = view.words.cast_mut();
    unsafe { std::ptr::copy_nonoverlapping(value, words.add(index * stride), stride) };
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
    pub const WINDOW: i32 = std::mem::offset_of!(Ctx, window) as i32;
    pub const SCALE: i32 = std::mem::offset_of!(Ctx, scale) as i32;
    pub const WIDTH: i32 = std::mem::offset_of!(Ctx, width) as i32;
    pub const HEIGHT: i32 = std::mem::offset_of!(Ctx, height) as i32;
    pub const DEPTH: i32 = std::mem::offset_of!(Ctx, depth) as i32;
    pub const ERROR: i32 = std::mem::offset_of!(Ctx, error) as i32;
    pub const VIEW_SIZE: i64 = std::mem::size_of::<View>() as i64;
    pub const VIEW_WIDTH: i32 = std::mem::offset_of!(View, width) as i32;
    pub const VIEW_HEIGHT: i32 = std::mem::offset_of!(View, height) as i32;
    pub const VIEW_DEPTH: i32 = std::mem::offset_of!(View, depth) as i32;
}
