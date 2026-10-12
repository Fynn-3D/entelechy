//! An element's failure from the device: the word a run writes, read once
//! the run is waited for.

use super::*;

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
    /// came before, not for what was submitted after.
    pub(crate) submitted: wgpu::SubmissionIndex,
    pub(crate) pile: Arc<Mutex<Vec<(wgpu::Buffer, wgpu::Buffer)>>>,
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
        let slice = back.slice(..);
        self.spent
            .time(|| {
                self.wait
                    .map_read_after(&self.device, &self.queue, slice, &self.submitted)
            })
            .map_err(ExecutionError::Device)?;
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
        if word == 0 && total != 0 {
            let element = u32::MAX - total;
            let width = self.totals_width.max(1);
            return Err(ExecutionError::Arithmetic {
                x: element % width,
                y: element / width,
                error: ArithmeticError::Overflow,
            });
        }
        if word == 0 {
            return entelechy_rules::Run::judge(
                self.run.work(),
                u64::from(blocks),
                u64::from(left),
            );
        }
        let word = u32::MAX - word;
        let element = word / 4;
        let width = self.width.max(1);
        let (x, y) = (element % width, element / width);
        Err(match word % 4 {
            wgsl::FAILED_BUDGET => ExecutionError::Exhausted {
                x,
                y,
                budget: self.budget,
            },
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
}

/// What the device said, in full. A validation error's `to_string` is
/// "Validation Error"; its description holds what failed and against which
/// limit.
/// Why a pipeline was not built: the shader the backend emitted was
/// refused (its own fault, said with the source), or the device's own
/// compiler failed on a shader that was valid.
#[derive(Debug)]
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
        let ours = wait.block_on(self.validation.pop());
        let short = wait.block_on(self.memory.pop());
        let theirs = wait.block_on(self.internal.pop());
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
}

pub(crate) fn device_error(error: &wgpu::Error) -> String {
    match error {
        wgpu::Error::Validation { description, .. } | wgpu::Error::Internal { description, .. } => {
            description.trim().to_owned()
        }
        other => other.to_string(),
    }
}

impl GpuBackend {
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
