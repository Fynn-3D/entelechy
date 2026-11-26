//! An element's failure from the device: the word a run writes, read once
//! the run is waited for.

use super::*;

/// Failure storage staged for a node: either a standalone `(word, back)` pair
/// (unbatched / measured / sliced runs) or a storage-aligned slot inside the
/// active frame batch's shared failure slab.
pub(crate) enum StagedFailure {
    Standalone(wgpu::Buffer, wgpu::Buffer),
    Batched {
        binding: crate::batch::FailureBinding,
        shared: Arc<crate::batch::BatchFailureReadback>,
    },
}

impl StagedFailure {
    /// Fixed-offset storage binding for non-dynamic helper passes (`prefix`,
    /// `reduce`, `scatter` fold/totals).
    pub(crate) fn binding(&self) -> wgpu::BindingResource<'_> {
        match self {
            Self::Standalone(word, _) => word.as_entire_binding(),
            Self::Batched { binding, .. } => binding.binding(),
        }
    }

    /// Base-zero storage binding and dynamic byte offset for `Pipeline.main`.
    pub(crate) fn dynamic_binding(&self) -> (wgpu::BindingResource<'_>, u32) {
        match self {
            Self::Standalone(word, _) => (
                wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: word,
                    offset: 0,
                    size: std::num::NonZeroU64::new(wgsl::FAILURE_BYTES),
                }),
                0,
            ),
            Self::Batched { binding, .. } => (binding.dynamic_binding(), binding.offset as u32),
        }
    }

    /// The underlying storage buffer for bind-group cache keying.
    pub(crate) fn buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Standalone(word, _) => word,
            Self::Batched { binding, .. } => &binding.buffer,
        }
    }
}

/// Evaluates the four failure words `(word, blocks, left, total)` into the
/// reference CPU error order: element failure, then whole-sum overflow, then
/// run budget exhaustion.
#[allow(clippy::too_many_arguments)]
pub(crate) fn judge_failure_words(
    word: u32,
    blocks: u32,
    left: u32,
    total: u32,
    width: u32,
    totals_width: u32,
    budget: u32,
    run: entelechy_rules::Run,
) -> Result<(), ExecutionError> {
    if word == 0 && total != 0 {
        let element = u32::MAX - total;
        let width = totals_width.max(1);
        return Err(ExecutionError::Arithmetic {
            x: element % width,
            y: element / width,
            error: ArithmeticError::Overflow,
        });
    }
    if word == 0 {
        return entelechy_rules::Run::judge(run.work(), u64::from(blocks), u64::from(left));
    }
    let word = u32::MAX - word;
    let element = word / 4;
    let width = width.max(1);
    let (x, y) = (element % width, element / width);
    Err(match word % 4 {
        wgsl::FAILED_BUDGET => ExecutionError::Exhausted { x, y, budget },
        wgsl::FAILED_DIVISION => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::DivisionByZero,
        },
        _ => ExecutionError::Arithmetic {
            x,
            y,
            error: ArithmeticError::Overflow,
        },
    })
}

/// Deferred failure check backed by one slot inside a batch's shared
/// [`crate::batch::BatchFailureReadback`].
pub(crate) struct BatchedFailureCheck {
    pub(crate) shared: Arc<crate::batch::BatchFailureReadback>,
    pub(crate) offset: u64,
    pub(crate) width: u32,
    pub(crate) totals_width: u32,
    pub(crate) budget: u32,
    pub(crate) run: entelechy_rules::Run,
}

impl Check for BatchedFailureCheck {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        let (word, blocks, left, total) = self.shared.read_slot(self.offset)?;
        judge_failure_words(
            word,
            blocks,
            left,
            total,
            self.width,
            self.totals_width,
            self.budget,
            self.run,
        )
    }
}

/// A device run's failure word: the first failing element, in row-major
/// order, and why. Read once the device is waited for.
pub(crate) struct FailureCheck {
    /// How the device is waited for, and the time it is counted in.
    pub(crate) wait: Wait,
    pub(crate) spent: Spent,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// The failure word and its readback buffer, back to the pile once read.
    pub(crate) pair: Option<(wgpu::Buffer, wgpu::Buffer)>,
    /// The run's submission: waiting for its word waits for it and what
    /// came before, not for what was submitted after. `None` when the run was
    /// batched and must flush the open frame batch first.
    pub(crate) submitted: Option<wgpu::SubmissionIndex>,
    /// Shared frame batcher flushed before mapping when `submitted` is `None`.
    pub(crate) batch: Arc<crate::batch::BatchHub>,
    /// The word's readback, asked for as the run was submitted where the
    /// device answers only for everything before ([`Wait::ask_map`]).
    pub(crate) asked: Option<crate::wait::Asked<Result<(), wgpu::BufferAsyncError>>>,
    pub(crate) pile: Arc<Mutex<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
    /// What the device said by itself ([`crate::told`]).
    pub(crate) heard: Arc<Mutex<Vec<String>>>,
    pub(crate) page_queue: crate::PageQueue,
    /// The mapped domain's width, which turns an element's number into the
    /// reference's column and row.
    pub(crate) width: u32,
    /// The width a whole sum totals over (a scatter's target, a prefix's
    /// rows), naming the element whose total does not fit.
    pub(crate) totals_width: u32,
    /// The run's budget, for naming a failure past it.
    pub(crate) budget: u32,
    /// The run's total, for naming a run past it.
    pub(crate) run: entelechy_rules::Run,
}

impl Check for FailureCheck {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let (word_buffer, back) = self.pair.take().expect("a check is waited on once");
        let submitted = match self.submitted.take() {
            Some(sub) => Some(sub),
            None => self.batch.flush_for_wait()?,
        };
        let slice = back.slice(..);
        let asked = self.asked.take();
        self.spent
            .time(|| match asked {
                Some(asked) => self
                    .wait
                    .answer(&self.device, asked)?
                    .map_err(|error| error.to_string()),
                None => match &submitted {
                    Some(sub) => self
                        .wait
                        .map_read_after(&self.device, &self.queue, slice, sub),
                    None => self.wait.map_read(&self.device, &self.queue, slice),
                },
            })
            .map_err(|error| ExecutionError::Device(crate::told(&self.heard, error)))?;
        self.page_queue.clear();
        let (word, blocks, left, total) = {
            let bytes = slice
                .get_mapped_range()
                .map_err(|error| ExecutionError::Device(error.to_string()))?;
            let at =
                |k: usize| u32::from_le_bytes([bytes[k], bytes[k + 1], bytes[k + 2], bytes[k + 3]]);
            (at(0), at(4), at(12), at(16))
        };
        back.unmap();
        self.pile
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push((word_buffer, back));
        // Named in order: an element's own failure; then the first element
        // whose whole sum does not fit, as the reference totals them; then,
        // past the run's total with no element failing, the run itself.
        judge_failure_words(
            word,
            blocks,
            left,
            total,
            self.width,
            self.totals_width,
            self.budget,
            self.run,
        )
    }
}

/// What the device said, in full. A validation error's `to_string` is
/// "Validation Error"; its description holds what failed and against which
/// limit.
/// Why a pipeline was not built: the shader the backend emitted was
/// refused (its own fault, said with the source), or the device's own
/// compiler failed on a shader that was valid.
#[derive(Clone, Debug)]
pub(crate) enum Unbuilt {
    Ours(String),
    Device(String),
}

impl Unbuilt {
    /// What it says, whoever's it is.
    pub(crate) fn said(self) -> String {
        match self {
            Self::Ours(said) | Self::Device(said) => said,
        }
    }
}

impl From<String> for Unbuilt {
    fn from(error: String) -> Self {
        Self::Ours(error)
    }
}

/// The device's errors caught while pipelines are built: a validation
/// error is the backend's own; an internal or out-of-memory one, the
/// device's, which would otherwise go uncaptured and panic.
pub(crate) struct Scopes {
    internal: wgpu::ErrorScopeGuard,
    memory: wgpu::ErrorScopeGuard,
    validation: wgpu::ErrorScopeGuard,
}

impl Scopes {
    pub(crate) fn push(device: &wgpu::Device) -> Self {
        let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
        let memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        Self {
            internal,
            memory,
            validation,
        }
    }

    /// What was caught since [`Scopes::push`], the backend's own fault said
    /// with the `source` it emitted.
    pub(crate) fn pop(self, wait: Wait, source: &str) -> Result<(), Unbuilt> {
        let validation = self.validation.pop();
        let memory = self.memory.pop();
        let internal = self.internal.pop();
        let ours = wait.block_on(validation);
        let short = wait.block_on(memory);
        let theirs = wait.block_on(internal);
        caught(ours, short, theirs, source)
    }

    /// Pops the scopes now and answers what they caught later, with the
    /// run's other checks: where the device answers by a round trip (a
    /// page), a frame then waits for none. The answers cannot leave the
    /// thread, so they wait on it ([`POPPED`]), by the check's number.
    pub(crate) fn pop_later(self, wait: Wait, spent: Spent) -> ScopeCheck {
        let popped: Popped = [
            Box::pin(self.validation.pop()),
            Box::pin(self.memory.pop()),
            Box::pin(self.internal.pop()),
        ];
        let number = POPPED.with(|all| {
            let mut all = all.borrow_mut();
            let number = all.0;
            all.0 += 1;
            all.1.insert(number, popped);
            number
        });
        ScopeCheck {
            number,
            wait,
            spent,
        }
    }
}

/// The answers of a run's scopes, in the order they were popped.
type Popped = [std::pin::Pin<Box<dyn std::future::Future<Output = Option<wgpu::Error>>>>; 3];

thread_local! {
    /// The scopes popped and not yet answered, by number, and the next number.
    static POPPED: std::cell::RefCell<(u64, std::collections::BTreeMap<u64, Popped>)> =
        std::cell::RefCell::default();
}

/// What a run's scopes caught, answered when the run is checked
/// ([`Scopes::pop_later`]).
pub(crate) struct ScopeCheck {
    number: u64,
    wait: Wait,
    /// The time its answers are waited for is counted in.
    spent: Spent,
}

impl Check for ScopeCheck {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        let Some([ours, short, theirs]) =
            POPPED.with(|all| all.borrow_mut().1.remove(&self.number))
        else {
            return Ok(());
        };
        let wait = self.wait;
        self.spent
            .time(|| {
                caught(
                    wait.block_on(ours),
                    wait.block_on(short),
                    wait.block_on(theirs),
                    "",
                )
            })
            .map_err(|error| ExecutionError::Device(error.said()))
    }
}

impl Drop for ScopeCheck {
    fn drop(&mut self) {
        POPPED.with(|all| all.borrow_mut().1.remove(&self.number));
    }
}

/// A run's checks, waited in order; the first failure is the answer.
pub(crate) struct Checks(pub(crate) Vec<Box<dyn Check>>);

impl Check for Checks {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        let mut first = Ok(());
        for check in self.0 {
            let answer = check.wait();
            if first.is_ok() {
                first = answer;
            }
        }
        first
    }
}

/// What a run's three scopes caught: a validation error is the backend's
/// own, said with the `source` it emitted; the rest, the device's.
fn caught(
    ours: Option<wgpu::Error>,
    short: Option<wgpu::Error>,
    theirs: Option<wgpu::Error>,
    source: &str,
) -> Result<(), Unbuilt> {
    if let Some(error) = ours {
        let said = device_error(&error);
        return Err(Unbuilt::Ours(if source.is_empty() {
            said
        } else {
            format!("{said}\n--- emitted WGSL ---\n{source}")
        }));
    }
    match short.or(theirs) {
        Some(error) => Err(Unbuilt::Device(device_error(&error))),
        None => Ok(()),
    }
}

pub(crate) fn device_error(error: &wgpu::Error) -> String {
    match error {
        wgpu::Error::Validation { description, .. } | wgpu::Error::Internal { description, .. } => {
            description.trim().to_owned()
        }
        other => other.to_string(),
    }
}

impl WgpuBackend {
    /// A failure record and its readback buffer, from the pile or new: the
    /// first failure's word, the blocks of work taken, and the most the run
    /// may take ([`wgsl::FAILURE_BYTES`]).
    pub(crate) fn failure_pair(&self) -> (wgpu::Buffer, wgpu::Buffer) {
        if let Some(pair) = self
            .failures
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .pop()
        {
            return pair;
        }
        let word = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("failure"),
            size: wgsl::FAILURE_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let back = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("failure back"),
            size: wgsl::FAILURE_BYTES,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (word, back)
    }
}
