//! An element's failure from the device: the word a run writes, read once
//! the run is waited for.

use super::*;

/// A device run's failure word: the first failing element, in row-major
/// order, and why. Read once the device is waited for.
pub(crate) struct FailureCheck {
    /// How the device is waited for.
    pub(crate) wait: Wait,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    /// The failure word and its readback buffer, back to the pile once read.
    pub(crate) pair: Option<(wgpu::Buffer, wgpu::Buffer)>,
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
    pub(crate) run: orla_rules::Run,
}

impl Check for FailureCheck {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let (word_buffer, back) = self.pair.take().expect("a check is waited on once");
        let slice = back.slice(..);
        self.wait
            .map_read(&self.device, &self.queue, slice)
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
            return orla_rules::Run::judge(self.run.work(), u64::from(blocks), u64::from(left));
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
