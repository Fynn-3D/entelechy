//! Preparing once: backends opened once for many sessions, each Kernel
//! prepared on them once. A run of many replays of one scene opens a
//! session each, fresh, its own runtime, Host and state, so no replay
//! sees another's; what they share is what preparing a Kernel makes of
//! its IR (a device's pipeline, the JIT's machine code), which is the
//! same for the same IR whichever session asks.
//!
//! [`Remembered`] wraps a backend and remembers what it prepared, by the
//! module: a module equal to one prepared before is handed the same
//! preparation. [`Warm`] holds the backends a run opened and gives each
//! session a runtime over them.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use entelechy_ir::Validated;
use entelechy_kernel::{Handle, Invocation, Runtime, Scope};
use entelechy_rules::backend::Checked;
use entelechy_rules::{Backend, ExecutionError, NativeOp, Prepared};

/// What was prepared, by Kernel id: each module prepared under it and
/// its preparation.
type Memo = Arc<Mutex<HashMap<String, Vec<(Validated, Arc<dyn Prepared>)>>>>;

/// A preparation handed out again: the backend that made it is handed
/// what it made.
struct Again(Arc<dyn Prepared>);

impl Prepared for Again {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// The preparation the inner backend made, under the one handed out.
fn made(prepared: &dyn Prepared) -> &dyn Prepared {
    match prepared.as_any().downcast_ref::<Again>() {
        Some(again) => again.0.as_ref(),
        None => prepared,
    }
}

/// A backend that prepares a module equal to one it prepared before by
/// handing out that preparation again. A refusal is not remembered.
pub struct Remembered {
    inner: Arc<dyn Backend>,
    memo: Memo,
}

impl Remembered {
    fn new(inner: Arc<dyn Backend>, memo: Memo) -> Self {
        Self { inner, memo }
    }
}

impl Backend for Remembered {
    fn name(&self) -> &'static str {
        self.inner.name()
    }
    fn declines(&self, module: &Validated) -> Option<String> {
        self.inner.declines(module)
    }
    fn finish(&self, prepared: &dyn Prepared) {
        self.inner.finish(made(prepared));
    }
    fn uploaded(&self) -> u64 {
        self.inner.uploaded()
    }
    fn repeats_exactly(&self) -> bool {
        self.inner.repeats_exactly()
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let id = module.manifest.id.to_string();
        let known = self.memo.lock().ok().and_then(|memo| {
            memo.get(&id)?
                .iter()
                .find(|(seen, _)| seen == module)
                .map(|(_, prepared)| Arc::clone(prepared))
        });
        if let Some(prepared) = known {
            return Ok(Box::new(Again(prepared)));
        }
        let prepared: Arc<dyn Prepared> = Arc::from(self.inner.prepare(module)?);
        if let Ok(mut memo) = self.memo.lock() {
            memo.entry(id)
                .or_default()
                .push((module.clone(), Arc::clone(&prepared)));
        }
        Ok(Box::new(Again(prepared)))
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner
            .execute(made(prepared), module, invocation, scope)
    }
    fn execute_checked(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Checked, ExecutionError> {
        self.inner
            .execute_checked(made(prepared), module, invocation, scope)
    }
    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        self.inner.execute_native(op, invocation, scope)
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
            .execute_repeated(made(prepared), module, invocation, scope, repeat, chain)
    }
    fn settle_device(&self) {
        self.inner.settle_device();
    }
    fn has_device(&self) -> bool {
        self.inner.has_device()
    }
    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u8>, String> {
        self.inner.download(device, byte_len)
    }
    fn submitted(&self) -> Option<Box<dyn entelechy_rules::Check>> {
        self.inner.submitted()
    }
    fn download_later(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Box<dyn entelechy_rules::Downloading>, String> {
        self.inner.download_later(device, byte_len)
    }
    fn same_resident(
        &self,
        a: &(dyn Any + Send + Sync),
        b: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        self.inner.same_resident(a, b, byte_len)
    }
    fn waited(&self) -> std::time::Duration {
        self.inner.waited()
    }
}

/// Backends opened once, and what was prepared on them, for many
/// sessions one after another: each [`Warm::runtime`] is a runtime of
/// its own over them.
pub struct Warm {
    main: Arc<dyn Backend>,
    ordered: Option<Arc<dyn Backend>>,
    main_memo: Memo,
    ordered_memo: Memo,
    limits: crate::Limits,
}

impl Warm {
    /// Over backends opened once (`Warm::open`, by the session's opening).
    pub(crate) fn over(
        main: Box<dyn Backend>,
        ordered: Option<Box<dyn Backend>>,
        limits: crate::Limits,
    ) -> Self {
        Self {
            main: Arc::from(main),
            ordered: ordered.map(Arc::from),
            main_memo: Memo::default(),
            ordered_memo: Memo::default(),
            limits,
        }
    }

    /// A runtime of its own over the backends, preparing each module once
    /// across every runtime given.
    pub fn runtime(&self) -> Runtime {
        self.runtime_through(&|backend| backend)
    }

    /// [`Warm::runtime`] with the Kernels [`crate::apart::chosen`] by
    /// `split` rounding apart ([`crate::apart`]): a module rounded apart
    /// is prepared once too, whichever runtime asks.
    pub fn runtime_apart(&self, split: u32) -> Runtime {
        self.runtime_through(&|backend| Box::new(crate::apart::Apart::new(backend, split)))
    }

    fn runtime_through(&self, wrap: &dyn Fn(Box<dyn Backend>) -> Box<dyn Backend>) -> Runtime {
        let main: Box<dyn Backend> = Box::new(Remembered::new(
            Arc::clone(&self.main),
            Arc::clone(&self.main_memo),
        ));
        let ordered = self.ordered.as_ref().map(|ordered| {
            Box::new(Remembered::new(
                Arc::clone(ordered),
                Arc::clone(&self.ordered_memo),
            )) as Box<dyn Backend>
        });
        let mut runtime = Runtime::with_backends(self.limits.words, wrap(main), ordered.map(wrap));
        runtime.host_mut().set_work(self.limits.work);
        runtime.set_small(self.limits.small);
        runtime
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    /// A backend that counts what it prepares and runs nothing.
    struct Counting(AtomicUsize);

    struct Mine;
    impl Prepared for Mine {
        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    impl Backend for Counting {
        fn name(&self) -> &'static str {
            "counting"
        }
        fn prepare(&self, _: &Validated) -> Result<Box<dyn Prepared>, String> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(Mine))
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

    fn module(scale: &str) -> Validated {
        entelechy_compiler::compile(&format!(
            "kernel t.k@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return float(element()) * {scale}; }} }}"
        ))
        .unwrap()
    }

    #[test]
    fn a_module_prepared_before_is_not_prepared_again_by_any_runtime() {
        let inner = Arc::new(Counting(AtomicUsize::new(0)));
        let memo = Memo::default();
        let first = Remembered::new(inner.clone(), Arc::clone(&memo));
        let second = Remembered::new(inner.clone(), Arc::clone(&memo));
        let one = first.prepare(&module("2.0")).unwrap();
        let again = second.prepare(&module("2.0")).unwrap();
        assert_eq!(inner.0.load(Ordering::Relaxed), 1);
        // The same id, another module: prepared.
        second.prepare(&module("3.0")).unwrap();
        assert_eq!(inner.0.load(Ordering::Relaxed), 2);
        // Each handed out preparation is the inner backend's own.
        assert!(made(one.as_ref()).as_any().downcast_ref::<Mine>().is_some());
        assert!(
            made(again.as_ref())
                .as_any()
                .downcast_ref::<Mine>()
                .is_some()
        );
    }

    #[test]
    fn a_remembered_device_marks_and_counts_its_waits_as_its_own() {
        let remembered = Remembered::new(Arc::new(Waiting), Memo::default());
        assert!(remembered.submitted().is_some());
        assert_eq!(remembered.waited(), std::time::Duration::from_millis(5));
    }
}
