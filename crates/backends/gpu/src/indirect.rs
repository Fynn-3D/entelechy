//! GPU-driven indirect compute dispatch for `Mode::Scatter` Kernels whose 1D
//! execution count comes from a device-resident `Int` Field (`domain = value(n)`),
//! keeping the count entirely on the GPU via `dispatch_workgroups_indirect` with
//! zero GPU->CPU readback stall.

use crate::failure::Scopes;
use crate::{WgpuBackend, wgsl};
use entelechy_rules::{Check, ExecutionError, IndirectCount};

const PREPARE_INDIRECT_SCATTER_WGSL: &str = "
struct IndirectSpec {
    index: u32,
    scale: i32,
    offset: i32,
    min_val: u32,
    ctx_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> spec: IndirectSpec;
@group(0) @binding(1) var<storage, read> count_buf: array<u32>;
@group(0) @binding(2) var<storage, read> ctx_tmpl: array<u32>;
@group(0) @binding(3) var<storage, read_write> ctx_out: array<u32>;
// Words 0..2: DispatchIndirectArgs (gx, gy, 1); word 3: non-zero when domain <= 0.
@group(0) @binding(4) var<storage, read_write> indirect_args: array<u32>;

@compute @workgroup_size(1, 1, 1)
fn main() {
    for (var i = 0u; i < spec.ctx_words; i += 1u) {
        ctx_out[i] = ctx_tmpl[i];
    }
    let raw = bitcast<i32>(count_buf[spec.index]);
    let val = raw * spec.scale + spec.offset;
    let clamped = select(val, max(val, i32(spec.min_val)), spec.min_val > 0u);
    let bad = spec.min_val == 0u && clamped <= 0i;
    let n = select(0u, u32(clamped), clamped > 0i);
    let groups = (n + 63u) / 64u;
    let gx = min(groups, 65535u);
    let safe_gx = max(gx, 1u);
    let gy = select(1u, (groups + safe_gx - 1u) / safe_gx, groups > 0u);
    // Patch ctx[0].x = w, ctx[0].y = 1, ctx[1].z = 1, ctx[5].y = fold (gx * 8), ctx[5].w = 0.
    ctx_out[0u] = n;
    ctx_out[1u] = 1u;
    ctx_out[6u] = 1u;
    ctx_out[21u] = gx * 8u;
    ctx_out[23u] = 0u;
    indirect_args[0u] = gx;
    indirect_args[1u] = gy;
    indirect_args[2u] = 1u;
    indirect_args[3u] = select(0u, 1u, bad);
}
";

pub(crate) struct IndirectScatterPipeline {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

pub(crate) fn build_indirect_scatter_pipeline(
    device: &wgpu::Device,
    wait: crate::Wait,
) -> Result<IndirectScatterPipeline, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy prepare indirect scatter"),
        source: wgpu::ShaderSource::Wgsl(PREPARE_INDIRECT_SCATTER_WGSL.into()),
    });
    let c = wgpu::ShaderStages::COMPUTE;
    let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: c,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy prepare indirect scatter"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, false),
            storage(4, false),
        ],
    });
    let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy prepare indirect scatter"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy prepare indirect scatter"),
        layout: Some(&pl),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = wait.block_on(guard.pop()) {
        return Err(format!(
            "indirect scatter pipeline: {}",
            crate::device_error(&error)
        ));
    }
    Ok(IndirectScatterPipeline { layout, pipeline })
}

pub(crate) fn spec_bytes(ic: IndirectCount, ctx_words: u32) -> [u8; 32] {
    let words: [u32; 8] = [
        ic.index,
        ic.scale as u32,
        ic.offset as u32,
        ic.min_val,
        ctx_words,
        0,
        0,
        0,
    ];
    let mut bytes = [0u8; 32];
    for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
        slot.copy_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Encodes the 1-thread GPU compute dispatch that patches `ctx` and computes
/// `indirect_args = [gx, gy, 1, bad]` from `count_buf`, then copies the patched
/// context into `ctx_uniform`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_prepare_indirect(
    device: &wgpu::Device,
    prep: &IndirectScatterPipeline,
    encoder: &mut wgpu::CommandEncoder,
    spec_slice: &crate::batch::UniformSlice,
    count_buf: &wgpu::Buffer,
    ctx_tmpl: &wgpu::Buffer,
    ctx_out: &wgpu::Buffer,
    ctx_uniform: &wgpu::Buffer,
    indirect_args: &wgpu::Buffer,
    ctx_bytes: u64,
    timing: Option<&crate::timing::Timing>,
) {
    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy prepare indirect scatter"),
        layout: &prep.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: spec_slice.binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: count_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: ctx_tmpl.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: ctx_out.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: indirect_args.as_entire_binding(),
            },
        ],
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("entelechy prepare indirect scatter"),
            timestamp_writes: timing.map(|t| t.compute(Some(0), None)),
        });
        pass.set_pipeline(&prep.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
    encoder.copy_buffer_to_buffer(ctx_out, 0, ctx_uniform, 0, ctx_bytes);
}

/// Staged check storage for an unclamped indirect scatter domain (`min_val == 0`).
pub(crate) enum StagedIndirectBad {
    Standalone(wgpu::Buffer),
    Batched {
        binding: crate::batch::FailureBinding,
        shared: std::sync::Arc<crate::batch::BatchFailureReadback>,
    },
}

/// Deferred frame-end check backed by one slot inside a batch's shared
/// [`crate::batch::BatchFailureReadback`], verifying an unclamped indirect
/// scatter count (`min_val == 0`) produced a positive 1D domain (`>= 1`).
pub(crate) struct BatchedIndirectDomainCheck {
    shared: std::sync::Arc<crate::batch::BatchFailureReadback>,
    offset: u64,
}

impl Check for BatchedIndirectDomainCheck {
    fn wait(self: Box<Self>) -> Result<(), ExecutionError> {
        if self.shared.read_u32(self.offset)? != 0 {
            return Err(ExecutionError::Device(
                "indirect domain resolved to a non-positive row count".into(),
            ));
        }
        Ok(())
    }
}

/// Deferred frame-end check verifying an unclamped indirect scatter count (`min_val == 0`)
/// produced a positive 1D domain (`>= 1`), preserving exact CPU domain-error semantics
/// without introducing any mid-frame GPU->CPU readback stall.
pub(crate) struct IndirectDomainCheck {
    wait: crate::Wait,
    spent: crate::Spent,
    device: wgpu::Device,
    queue: wgpu::Queue,
    staging: Option<wgpu::Buffer>,
    submitted: Option<wgpu::SubmissionIndex>,
    asked: Option<crate::wait::Asked<Result<(), wgpu::BufferAsyncError>>>,
    batch: std::sync::Arc<crate::batch::BatchHub>,
    spare: crate::pool::Spare,
    heard: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
    page_queue: crate::PageQueue,
}

impl Drop for IndirectDomainCheck {
    fn drop(&mut self) {
        if let Some(staging) = self.staging.take() {
            if self.asked.is_some() {
                crate::wait::discard(staging);
            } else {
                self.spare.keep(staging, None);
            }
        }
    }
}

impl Check for IndirectDomainCheck {
    fn wait(mut self: Box<Self>) -> Result<(), ExecutionError> {
        let Some(staging) = self.staging.take() else {
            return Ok(());
        };
        let submitted = match self.submitted.take() {
            Some(sub) => Some(sub),
            None => self.batch.flush_for_wait()?,
        };
        let asked = self.asked.take();
        self.spent
            .time(|| match asked {
                Some(asked) => match &submitted {
                    Some(sub) => self
                        .wait
                        .answer_after(&self.device, sub, asked)?
                        .map_err(|e| e.to_string()),
                    None => self
                        .wait
                        .answer(&self.device, asked)?
                        .map_err(|e| e.to_string()),
                },
                None => match &submitted {
                    Some(sub) => self.wait.map_read_after(
                        &self.device,
                        &self.queue,
                        staging.slice(..16),
                        sub,
                    ),
                    None => self
                        .wait
                        .map_read(&self.device, &self.queue, staging.slice(..16)),
                },
            })
            .map_err(|e| ExecutionError::Device(crate::told(&self.heard, e)))?;
        self.page_queue.clear();
        let bad = {
            let view = staging
                .slice(..16)
                .get_mapped_range()
                .map_err(|e| ExecutionError::Device(e.to_string()))?;
            u32::from_le_bytes([view[0], view[1], view[2], view[3]]) != 0
        };
        staging.unmap();
        self.spare.keep(staging, None);
        if bad {
            return Err(ExecutionError::Device(
                "indirect domain resolved to a non-positive row count".into(),
            ));
        }
        Ok(())
    }
}

impl WgpuBackend {
    pub(crate) fn supports_indirect_scatter_impl(&self, module: &entelechy_ir::Validated) -> bool {
        matches!(module.manifest.mode, entelechy_ir::Mode::Scatter { .. })
            && !wgsl::emitted_pairs(module).is_empty()
            && wgsl::scatter_plan(module) != wgsl::ScatterPlan::Sorted
    }

    pub(crate) fn execute_indirect_scatter_impl(
        &self,
        prepared: &dyn entelechy_rules::Prepared,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_abi::Invocation,
        scope: &mut entelechy_abi::Scope<'_>,
        indirect: IndirectCount,
    ) -> Option<Result<entelechy_rules::backend::Checked, ExecutionError>> {
        if !self.supports_indirect_scatter_impl(module) {
            return None;
        }
        let pipeline = match self.built_for(prepared, module, invocation, scope) {
            Ok(p) => p,
            Err(e) => return Some(Err(e)),
        };
        if !pipeline.emits {
            return None;
        }
        let prep = match self
            .indirect_scatter
            .get_or_init(|| build_indirect_scatter_pipeline(&self.device, self.wait))
        {
            Ok(p) => p,
            Err(e) => return Some(Err(ExecutionError::Device(e.clone()))),
        };
        Some(self.run_indirect_scatter(prep, pipeline, module, invocation, scope, indirect))
    }

    fn run_indirect_scatter(
        &self,
        prep: &IndirectScatterPipeline,
        pipeline: &crate::prepare::Pipeline,
        module: &entelechy_ir::Validated,
        invocation: &entelechy_abi::Invocation,
        scope: &mut entelechy_abi::Scope<'_>,
        indirect: IndirectCount,
    ) -> Result<entelechy_rules::backend::Checked, ExecutionError> {
        let entelechy_ir::Mode::Scatter { base, .. } = module.manifest.mode else {
            return Err(ExecutionError::Device(
                "indirect scatter called on non-scatter mode".into(),
            ));
        };
        entelechy_rules::check_scatter(invocation, scope, base as usize)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let encoding = self.waiter_control.begin_encoding();
        let output_domain = scope.size(invocation.inputs[base as usize])?;
        crate::stage::outputs_fit(module, output_domain, scope)?;
        let timing_set = self.timing_set(2);
        let allow_batch = self.batch.can_batch(self.measured(), &self.frames);
        let staged = self.stage_with_indirect(
            pipeline,
            module,
            invocation,
            scope,
            entelechy_abi::Domain::Rows(1),
            timing_set.as_ref(),
            usize::MAX,
            allow_batch,
            Some((prep, indirect)),
        )?;
        let crate::stage::Staged {
            mut encoder,
            batched,
            bind_group: _,
            dynamic_offsets: _,
            dynamic_offset_len: _,
            further_slices: _,
            context_buffer: _,
            context_slice,
            params_slice: _,
            input_buffers,
            bound_inputs,
            bound_textures: _,
            mut temp_buffers,
            memory_outputs,
            output_buffer,
            further_outputs,
            output_words,
            element_count,
            failure,
            target,
            highs,
            indirect_bad_check,
        } = staged;
        let final_buffer = if let Some(scatter) = pipeline.scatter.as_ref() {
            self.combine(
                &mut encoder,
                scatter,
                (target, highs),
                failure.as_ref(),
                input_buffers[base as usize]
                    .as_ref()
                    .expect("scatter base holds a storage buffer"),
                &context_slice,
                output_buffer,
                output_domain.count(),
                element_count,
                output_words,
                timing_set.as_ref(),
                &mut temp_buffers,
            )
        } else {
            output_buffer
        };
        let total_ops = (output_domain.count() as u64).saturating_mul(64);
        let (submitted, scoped) = if batched {
            if let Some(crate::StagedFailure::Standalone(word, back)) = &failure {
                encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
            }
            let _ = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
            self.frames.note_batched_run();
            let mut keep_buffers: Vec<wgpu::Buffer> = bound_inputs
                .iter()
                .filter(|(binding, _)| wgsl::is_packed_binding(*binding))
                .map(|(_, buffer)| buffer.clone())
                .collect();
            keep_buffers.extend(temp_buffers);
            let page_budget = self
                .wait
                .parks_with_host()
                .then(|| self.page_pace_ops_budget());
            drop(encoding);
            self.batch
                .finish_node(encoder, total_ops, keep_buffers, Vec::new(), page_budget)?;
            (None, None)
        } else {
            let scopes = Scopes::push(&self.device);
            if let Some(crate::StagedFailure::Standalone(word, back)) = &failure {
                encoder.copy_buffer_to_buffer(word, 0, back, 0, wgsl::FAILURE_BYTES);
            }
            let timing = self.resolve_timing(&mut encoder, timing_set.as_ref(), 2);
            let sub = self.submit(encoder.finish());
            if self.wait.parks_with_host() && !self.measured() {
                self.page_queue.note(total_ops);
            }
            drop(encoding);
            self.settle_measured(sub.clone(), timing);
            for (binding, buffer) in &bound_inputs {
                if wgsl::is_packed_binding(*binding) {
                    self.spare.keep(buffer.clone(), None);
                }
            }
            for buffer in temp_buffers {
                self.spare.keep(buffer, None);
            }
            let scoped = if self.wait.parks_with_host() && !self.measured() {
                Some(Box::new(scopes.pop_later(self.wait, self.spent.clone())) as Box<dyn Check>)
            } else {
                scopes
                    .pop(self.wait, "")
                    .map_err(|error| ExecutionError::Device(error.said()))?;
                None
            };
            (Some(sub), scoped)
        };
        let outputs = crate::stage::adopt_outputs(
            scope,
            module,
            output_domain,
            final_buffer,
            further_outputs,
            memory_outputs,
            Some(self.spare.clone()),
        )?;
        let mut checks: Vec<Box<dyn Check>> = Vec::new();
        if let Some(s) = scoped {
            checks.push(s);
        }
        if let Some(staged_failure) = failure {
            match staged_failure {
                crate::StagedFailure::Batched { binding, shared } => {
                    checks.push(Box::new(crate::BatchedFailureCheck {
                        shared,
                        offset: binding.offset,
                        width: 1,
                        totals_width: output_domain.width(),
                        budget: module.manifest.budget,
                        run: entelechy_rules::Run::new(scope.work()),
                    }));
                }
                crate::StagedFailure::Standalone(word, back) => {
                    let asked = if submitted.is_some() {
                        self.wait.ask_map(&self.queue, back.slice(..))
                    } else {
                        None
                    };
                    checks.push(Box::new(crate::FailureCheck {
                        asked,
                        wait: self.wait,
                        spent: self.spent.clone(),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        pair: Some((word, back)),
                        submitted: submitted.clone(),
                        batch: std::sync::Arc::clone(&self.batch),
                        pile: std::sync::Arc::clone(&self.failures),
                        heard: std::sync::Arc::clone(&self.heard),
                        page_queue: self.page_queue.clone(),
                        width: 1,
                        totals_width: output_domain.width(),
                        budget: module.manifest.budget,
                        run: entelechy_rules::Run::new(scope.work()),
                    }));
                }
            }
        }
        if let Some(bad) = indirect_bad_check {
            match bad {
                StagedIndirectBad::Batched { binding, shared } => {
                    checks.push(Box::new(BatchedIndirectDomainCheck {
                        shared,
                        offset: binding.offset,
                    }));
                }
                StagedIndirectBad::Standalone(bad_staging) => {
                    let asked = if submitted.is_some() {
                        self.wait.ask_map(&self.queue, bad_staging.slice(..16))
                    } else {
                        None
                    };
                    checks.push(Box::new(IndirectDomainCheck {
                        wait: self.wait,
                        spent: self.spent.clone(),
                        device: self.device.clone(),
                        queue: self.queue.clone(),
                        staging: Some(bad_staging),
                        submitted,
                        asked,
                        batch: std::sync::Arc::clone(&self.batch),
                        spare: self.spare.clone(),
                        heard: std::sync::Arc::clone(&self.heard),
                        page_queue: self.page_queue.clone(),
                    }));
                }
            }
        }
        let check = match checks.len() {
            0 => None,
            1 => checks.pop(),
            _ => Some(Box::new(crate::Checks(checks)) as Box<dyn Check>),
        };
        Ok((outputs, check))
    }
}
