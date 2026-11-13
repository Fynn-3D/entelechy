//! Rounding apart: a scene run with some of its Kernels rounding as
//! another device might, to find a decision two Kernels must agree on and
//! each makes for itself. Devices round an expression to different last
//! bits in different Kernels, as their compilers fuse `a * b + c` into one
//! rounding or take `a / b` as `a * (1 / b)`; a replay allows what that
//! changes in a picture, so what fails when some Kernels round apart is a
//! value two Kernels compute each and must hold equal. It is made in one
//! ([decision](../../../../docs/DECISIONS.md)).
//!
//! [`SPLITS`] ways of choosing the Kernels, each a bit of a hash of the
//! Kernel's id, so a pair that one way leaves together another parts.

use std::sync::atomic::{AtomicUsize, Ordering};

use entelechy_ir::{
    BinaryOp, Function, Instruction, Intrinsic, Op, Type, Validated, Value, ValueId,
};
use entelechy_kernel::{Handle, Invocation, Scope};
use entelechy_rules::backend::Checked;
use entelechy_rules::{Backend, ExecutionError, NativeOp, Prepared};

/// The ways of choosing which Kernels round apart: a run each.
pub const SPLITS: u32 = 3;

/// Whether Kernel `id` rounds apart in split `split`: bit `split` of a
/// hash of its id (FNV-1a), the same wherever it is asked.
pub fn chosen(id: &str, split: u32) -> bool {
    let hash = id.bytes().fold(0x811c_9dc5u32, |hash, byte| {
        (hash ^ u32::from(byte)).wrapping_mul(0x0100_0193)
    });
    hash >> (split % 32) & 1 == 1
}

/// `module` rounding as another device might: every float `a * b + c`
/// fused into one `fma`, rounded once, and every float `a / b` taken as
/// `a * (1 / b)`, rounded twice; and how many operations changed.
pub fn rounded_apart(module: &Validated) -> Result<(Validated, usize), String> {
    let mut changed = 0;
    let mut module = module.clone().into_module();
    let functions = module
        .functions
        .iter_mut()
        .chain(std::iter::once(&mut module.entry))
        .chain(module.merge.iter_mut())
        .chain(module.init.iter_mut());
    for function in functions {
        let mut types = std::collections::HashMap::new();
        typed(&function.body, &mut types);
        let mut products = std::collections::HashMap::new();
        multiplied(&function.body, &mut products);
        fuse(&mut function.body, &products, &types, &mut changed);
        // New values take numbers from the top; the function is numbered
        // again as a module is.
        let mut next = u32::MAX;
        reciprocal(&mut function.body, &mut next, &mut changed);
        *function = entelechy_ir::renumbered(std::mem::replace(function, empty()));
    }
    entelechy_ir::validate(module)
        .map(|module| (module, changed))
        .map_err(|error| error.to_string())
}

fn float(ty: Type) -> bool {
    matches!(ty, Type::Float | Type::Vec2 | Type::Vec3 | Type::Vec4)
}

fn typed(body: &[Instruction], types: &mut std::collections::HashMap<ValueId, Type>) {
    for instruction in body {
        types.insert(instruction.id, instruction.ty);
        for block in instruction.op.blocks() {
            typed(&block.body, types);
        }
    }
}

fn multiplied(
    body: &[Instruction],
    found: &mut std::collections::HashMap<ValueId, (ValueId, ValueId)>,
) {
    for instruction in body {
        if let Op::Binary(BinaryOp::Mul, a, b) = instruction.op
            && float(instruction.ty)
        {
            found.insert(instruction.id, (a, b));
        }
        for block in instruction.op.blocks() {
            multiplied(&block.body, found);
        }
    }
}

fn fuse(
    body: &mut [Instruction],
    products: &std::collections::HashMap<ValueId, (ValueId, ValueId)>,
    types: &std::collections::HashMap<ValueId, Type>,
    changed: &mut usize,
) {
    for instruction in body.iter_mut() {
        if let Op::Binary(BinaryOp::Add, x, y) = instruction.op
            && float(instruction.ty)
        {
            let ty = instruction.ty;
            let same = |v: &ValueId| types.get(v) == Some(&ty);
            let pick = |p: ValueId, other: ValueId| {
                products
                    .get(&p)
                    .filter(|(a, b)| same(a) && same(b) && same(&other))
                    .map(|(a, b)| (*a, *b, other))
            };
            if let Some((a, b, c)) = pick(x, y).or_else(|| pick(y, x)) {
                instruction.op = Op::Intrinsic(Intrinsic::Fma, vec![a, b, c]);
                *changed += 1;
            }
        }
        for block in instruction.op.blocks_mut() {
            fuse(&mut block.body, products, types, changed);
        }
    }
}

fn reciprocal(body: &mut Vec<Instruction>, next: &mut u32, changed: &mut usize) {
    let mut out = Vec::with_capacity(body.len());
    for mut instruction in body.drain(..) {
        for block in instruction.op.blocks_mut() {
            reciprocal(&mut block.body, next, changed);
        }
        if let Op::Binary(BinaryOp::Div, a, b) = instruction.op
            && instruction.ty == Type::Float
        {
            let one = ValueId(*next);
            let inverse = ValueId(*next - 1);
            *next -= 2;
            out.push(Instruction {
                id: one,
                ty: Type::Float,
                op: Op::Const(Value::Float(1.0)),
            });
            out.push(Instruction {
                id: inverse,
                ty: Type::Float,
                op: Op::Binary(BinaryOp::Div, one, b),
            });
            instruction.op = Op::Binary(BinaryOp::Mul, a, inverse);
            *changed += 1;
        }
        out.push(instruction);
    }
    *body = out;
}

fn empty() -> Function {
    Function {
        name: String::new(),
        args: Vec::new(),
        body: Vec::new(),
        results: Vec::new(),
        bindings: Vec::new(),
    }
}

/// A backend whose chosen Kernels are prepared rounding apart; the rest,
/// and everything it runs, are its inner backend's.
pub struct Apart {
    inner: Box<dyn Backend>,
    split: u32,
    changed: AtomicUsize,
}

impl Apart {
    /// `inner`, the Kernels [`chosen`] by `split` rounding apart.
    pub fn new(inner: Box<dyn Backend>, split: u32) -> Self {
        Self {
            inner,
            split,
            changed: AtomicUsize::new(0),
        }
    }

    /// How many operations the Kernels prepared so far round apart.
    pub fn changed(&self) -> usize {
        self.changed.load(Ordering::Relaxed)
    }
}

impl Backend for Apart {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        // What the inner backend prepared is the module rounded apart.
        if chosen(&module.manifest.id.to_string(), self.split)
            && let Ok((apart, _)) = rounded_apart(module)
        {
            return self.inner.declines(&apart);
        }
        self.inner.declines(module)
    }
    fn finish(&self, prepared: &dyn Prepared) {
        self.inner.finish(prepared);
    }
    fn finished(&self, prepared: &dyn Prepared) -> bool {
        self.inner.finished(prepared)
    }
    fn uploaded(&self) -> u64 {
        self.inner.uploaded()
    }
    fn upload(&self, handle: Handle, scope: &mut Scope<'_>) -> Result<bool, ExecutionError> {
        self.inner.upload(handle, scope)
    }
    fn join(
        &self,
        parts: &[(Handle, std::ops::Range<usize>)],
        schema: &entelechy_kernel::Schema,
        domain: entelechy_kernel::Domain,
        scope: &mut Scope<'_>,
    ) -> Result<Option<Handle>, ExecutionError> {
        self.inner.join(parts, schema, domain, scope)
    }
    fn repeats_exactly(&self) -> bool {
        self.inner.repeats_exactly()
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        if !chosen(&module.manifest.id.to_string(), self.split) {
            return self.inner.prepare(module);
        }
        let (apart, changed) = rounded_apart(module)?;
        self.changed.fetch_add(changed, Ordering::Relaxed);
        self.inner.prepare(&apart)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner.execute(prepared, module, invocation, scope)
    }
    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        self.inner
            .execute_checked(prepared, module, invocation, scope)
    }
    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner.execute_native(op, invocation, scope)
    }
    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        self.inner
            .execute_repeated(prepared, module, invocation, scope, repeat, chain)
    }
    fn settle_device(&self) {
        self.inner.settle_device();
    }
    fn has_device(&self) -> bool {
        self.inner.has_device()
    }
    fn download(
        &self,
        device: &(dyn std::any::Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, String> {
        self.inner.download(device, byte_len)
    }
    fn last_device_time(&self) -> Option<std::time::Duration> {
        self.inner.last_device_time()
    }
    fn measurement_floor(&self) -> std::time::Duration {
        self.inner.measurement_floor()
    }
    fn set_measured(&self, on: bool) {
        self.inner.set_measured(on);
    }
    fn measures_the_device(&self) -> bool {
        self.inner.measures_the_device()
    }
    fn set_clocked(&self, on: bool) {
        self.inner.set_clocked(on);
    }
    fn clock_mark(&self) -> usize {
        self.inner.clock_mark()
    }
    fn end_frame(&self) -> Option<u64> {
        self.inner.end_frame()
    }
    fn clocked_frame(&self) -> Option<entelechy_rules::ClockedFrame> {
        self.inner.clocked_frame()
    }
    fn memory(&self) -> Vec<(&'static str, u64)> {
        self.inner.memory()
    }
    fn evaluation_begun(&self) {
        self.inner.evaluation_begun();
    }
    fn submitted(&self) -> Option<Box<dyn entelechy_rules::Check>> {
        self.inner.submitted()
    }
    fn download_later(
        &self,
        device: &(dyn std::any::Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Box<dyn entelechy_rules::Downloading>, String> {
        self.inner.download_later(device, byte_len)
    }
    fn same_resident(
        &self,
        a: &(dyn std::any::Any + Send + Sync),
        b: &(dyn std::any::Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        self.inner.same_resident(a, b, byte_len)
    }
    fn waited(&self) -> std::time::Duration {
        self.inner.waited()
    }
}

#[cfg(all(test, feature = "documents"))]
mod tests {
    use super::*;

    /// A device that has waited five milliseconds and marks what it was
    /// given.
    struct Waiting;

    struct Done;
    impl entelechy_rules::Check for Done {
        fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
            Ok(())
        }
    }

    impl Backend for Waiting {
        fn name(&self) -> &'static str {
            "waiting"
        }
        fn has_device(&self) -> bool {
            true
        }
        fn submitted(&self) -> Option<Box<dyn entelechy_rules::Check>> {
            Some(Box::new(Done))
        }
        fn waited(&self) -> std::time::Duration {
            std::time::Duration::from_millis(5)
        }
        fn prepare(&self, _: &Validated) -> Result<Box<dyn Prepared>, String> {
            Err("prepares nothing".into())
        }
        fn execute(
            &self,
            _: &dyn Prepared,
            _: &Validated,
            _: &Invocation,
            _: &mut Scope<'_>,
        ) -> Result<Vec<Handle>, ExecutionError> {
            Err(ExecutionError::Device("runs nothing".into()))
        }
    }

    fn compiled(source: &str) -> Validated {
        entelechy_compiler::compile(source).unwrap()
    }

    #[test]
    fn a_product_added_is_fused_and_a_quotient_taken_by_its_reciprocal() {
        let module = compiled(
            "kernel t.k@1 { requires { field.read; field.write; } param a: Float = 0.1; param b: Float = 0.7; output result: Rows<Float>; compute { return a * b + float(element()) / b; } }",
        );
        let (apart, changed) = rounded_apart(&module).unwrap();
        assert!(changed >= 2, "{changed}");
        let ops: Vec<String> = apart
            .entry
            .body
            .iter()
            .map(|i| format!("{:?}", i.op))
            .collect();
        assert!(ops.iter().any(|op| op.contains("Fma")), "{ops:?}");
        assert!(
            !ops.iter().any(|op| op.starts_with("Binary(Add")),
            "{ops:?}"
        );
    }

    #[test]
    fn a_device_rounded_apart_marks_and_counts_its_waits_as_its_own() {
        let apart = Apart::new(Box::new(Waiting), 1);
        assert!(apart.submitted().is_some());
        assert_eq!(apart.waited(), std::time::Duration::from_millis(5));
    }
}
