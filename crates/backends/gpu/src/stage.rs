//! Bringing a run's inputs and parameters to the device and binding
//! them: what is uploaded, what stays resident, and the extent a run
//! covers and writes.

use super::*;

/// The most words written to the device from one copy of them as bytes:
/// a large Field goes up in pieces, never copied whole.
const WRITE_WORDS: usize = 1 << 18;

/// Upload scratch capacity kept warm between dispatches (`256 KiB`).
const KEPT_UPLOAD_BYTES: usize = 256 * 1024;

/// Reusable staging buffers owned by [`GpuBackend`] and used under
/// [`GpuBackend::lock`].
#[derive(Default)]
pub(crate) struct StageScratch {
    context: Vec<u8>,
    params: Vec<u8>,
    words: Vec<u32>,
    upload: Vec<u8>,
}

impl WgpuBackend {
    /// Writes `words` to `buffer` from word `at` on, queued for the next
    /// submission; the queue copies each piece as it is written.
    pub(crate) fn write_words(&self, buffer: &wgpu::Buffer, at: usize, words: &[u32]) {
        if words.is_empty() {
            return;
        }
        let mut scratch = self
            .staging
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for (k, piece) in words.chunks(WRITE_WORDS).enumerate() {
            let byte_len = piece.len() * 4;
            scratch.upload.resize(byte_len, 0);
            for (dst, &word) in scratch.upload.chunks_exact_mut(4).zip(piece) {
                dst.copy_from_slice(&word.to_le_bytes());
            }
            self.uploaded
                .fetch_add(byte_len as u64, std::sync::atomic::Ordering::Relaxed);
            let offset = (at + k * WRITE_WORDS) as u64 * 4;
            self.queue.write_buffer(buffer, offset, &scratch.upload);
        }
        if scratch.upload.capacity() > KEPT_UPLOAD_BYTES {
            scratch.upload = Vec::new();
        }
    }
}

fn fill_param_bytes(params: &[Value], words: &mut Vec<u32>, bytes: &mut Vec<u8>) {
    // Each parameter's words from a `vec4<u32>` slot of its own, as many
    // slots as they fill (`wgsl::param_slots`).
    bytes.clear();
    bytes.reserve(params.len().max(1) * 16);
    for value in params {
        words.clear();
        value.push_words(words);
        let padded = words.len().div_ceil(4).max(1) * 4;
        words.resize(padded, 0);
        for &word in words.iter() {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    if bytes.is_empty() {
        bytes.resize(16, 0);
    }
}

#[allow(dead_code)]
pub(crate) fn param_bytes(params: &[Value]) -> Vec<u8> {
    let mut words = Vec::new();
    let mut bytes = Vec::new();
    fill_param_bytes(params, &mut words, &mut bytes);
    bytes
}

/// What the map pass leaves ready for the mode's remaining passes.
pub(crate) struct Staged {
    pub(crate) encoder: wgpu::CommandEncoder,
    pub(crate) bind_group: wgpu::BindGroup,
    pub(crate) further_slices: Vec<(u32, u32, u32, u32, u32)>,
    pub(crate) context_buffer: wgpu::Buffer,
    pub(crate) params_buffer: wgpu::Buffer,
    pub(crate) input_buffers: Vec<wgpu::Buffer>,
    /// What the passes bind for the inputs, by binding: the inputs' own
    /// buffers, and the pack where some are packed.
    pub(crate) bound_inputs: Vec<(u32, wgpu::Buffer)>,
    /// A memory's copy, written by the steps, to adopt as that output.
    pub(crate) memory_outputs: Vec<(usize, wgpu::Buffer, entelechy_abi::Domain)>,
    pub(crate) output_buffer: wgpu::Buffer,
    /// A map's outputs after the first, in the manifest's order.
    pub(crate) further_outputs: Vec<wgpu::Buffer>,
    pub(crate) output_words: usize,
    pub(crate) element_count: usize,
    /// The word the passes record their first failure in, when the module can
    /// fail, and its readback buffer.
    pub(crate) failure: Option<(wgpu::Buffer, wgpu::Buffer)>,
    /// A scatter's target, ready before the entry when the entry emits.
    pub(crate) target: Option<wgpu::Buffer>,
    /// The high words beside it, where it sums whole
    /// ([`wgsl::sums_whole`]).
    pub(crate) highs: Option<wgpu::Buffer>,
}

/// The domain the entry maps over: a reduction's and a scatter's first
/// input, a scan's rows (checked), else the invocation's.
pub(crate) fn execution_domain(
    module: &Validated,
    invocation: &Invocation,
    scope: &Scope<'_>,
) -> Result<entelechy_abi::Domain, ExecutionError> {
    Ok(match module.manifest.mode {
        Mode::Reduce | Mode::Prefix { .. } => scope.size(invocation.inputs[0])?,
        Mode::Scan => entelechy_rules::check_scan(invocation, scope)?,
        Mode::Scatter { base, .. } => {
            entelechy_rules::check_scatter(invocation, scope, base as usize)?;
            invocation.domain
        }
        Mode::Map | Mode::Group { .. } => invocation.domain,
    })
}

/// Whether the Host budget has room for every output over `domain`, asked
/// before a buffer is made for one.
pub(crate) fn outputs_fit(
    module: &Validated,
    domain: entelechy_abi::Domain,
    scope: &Scope<'_>,
) -> Result<(), ExecutionError> {
    if module.manifest.outputs.len() == 1 && !wgsl::highs_beside(module) {
        return Ok(scope.can_hold(&[(&module.manifest.outputs[0].ty.schema, domain)])?);
    }
    let mut fields: Vec<_> = module
        .manifest
        .outputs
        .iter()
        .map(|port| (&port.ty.schema, domain))
        .collect();
    // A whole sum keeps a high word beside each word of what it totals.
    if wgsl::highs_beside(module) {
        fields.push(fields[0]);
    }
    Ok(scope.can_hold(&fields)?)
}

/// The output and the memories as the scope's resident Fields; on a
/// failure, whatever was adopted is released.
pub(crate) fn adopt_outputs(
    scope: &mut Scope<'_>,
    module: &Validated,
    output_domain: entelechy_abi::Domain,
    final_buffer: wgpu::Buffer,
    further_outputs: Vec<wgpu::Buffer>,
    memory_outputs: Vec<(usize, wgpu::Buffer, entelechy_abi::Domain)>,
    spare: Option<Spare>,
) -> Result<Vec<Handle>, ExecutionError> {
    let output_port = &module.manifest.outputs[0];
    let device = Box::new(DeviceImage {
        buffer: final_buffer,
        spare: spare.clone(),
        holds: None,
    });
    let output = scope.adopt_resident(output_port.ty.schema.clone(), output_domain, device)?;
    let mut outputs = vec![output];
    // A map's further outputs over the same domain, then the memories,
    // which is the manifest's order.
    let further = module
        .manifest
        .outputs
        .iter()
        .skip(1)
        .zip(further_outputs)
        .map(|(port, buffer)| (port.ty.schema.clone(), output_domain, buffer));
    let memories = memory_outputs.into_iter().map(|(index, buffer, extent)| {
        (
            module.manifest.inputs[index].ty.schema.clone(),
            extent,
            buffer,
        )
    });
    for (schema, extent, buffer) in further.chain(memories) {
        let adopted = scope.adopt_resident(
            schema,
            extent,
            Box::new(DeviceImage {
                buffer,
                spare: spare.clone(),
                holds: None,
            }),
        );
        match adopted {
            Ok(handle) => outputs.push(handle),
            Err(error) => {
                for handle in outputs {
                    let _ = scope.release(handle);
                }
                return Err(error.into());
            }
        }
    }
    Ok(outputs)
}

impl WgpuBackend {
    pub(crate) fn bind_input(
        &self,
        scope: &mut Scope<'_>,
        handle: Handle,
    ) -> Result<wgpu::Buffer, ExecutionError> {
        let resident = scope
            .resident(handle)?
            .and_then(|device| device.downcast_ref::<DeviceImage>())
            .map(|device| device.buffer.clone());
        if let Some(buffer) = resident {
            return Ok(buffer);
        }
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let len = scope.storage(handle)?.len();
        let size = (len.max(4) as u64 * 4).div_ceil(16) * 16;
        // A Field made from words a buffer here still holds, or holds but
        // on pages of its own, is brought up from that buffer on the words
        // either may differ on alone; else from a copy of the buffer its
        // base lives in. A write is queued for the next submission, after
        // every command that read the buffer before.
        let made = scope.made_from(handle)?;
        let found = self.spare.take_made(&made, size).or_else(|| {
            made.iter().find_map(|from| {
                self.copied(scope, from.words, size, usage)
                    .map(|buffer| (buffer, from.changed.clone()))
            })
        });
        let (buffer, spans) = match found {
            Some(found) => found,
            None => (
                self.spare.take(&self.device, size, usage),
                std::iter::once(0..len).collect(),
            ),
        };
        let storage = scope.storage(handle)?;
        for span in spans {
            self.write_words(&buffer, span.start, &storage[span]);
        }
        let holds = Held {
            words: scope.version(handle)?,
            made,
        };
        scope.make_resident(
            handle,
            Box::new(DeviceImage {
                buffer: buffer.clone(),
                spare: Some(self.spare.clone()),
                holds: Some(holds),
            }),
        )?;
        Ok(buffer)
    }

    /// A copy, on the device, of the buffer a resident Field holding
    /// `words` has, when it holds `size` bytes or more.
    fn copied(
        &self,
        scope: &Scope<'_>,
        words: Version,
        size: u64,
        usage: wgpu::BufferUsages,
    ) -> Option<wgpu::Buffer> {
        if !scope.holds(words).ok()? {
            return None;
        }
        let from = scope
            .resident(words.field())
            .ok()??
            .downcast_ref::<DeviceImage>()?
            .buffer
            .clone();
        if from.size() < size {
            return None;
        }
        let to = self.spare.take(&self.device, size, usage);
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("made from"),
            });
        encoder.copy_buffer_to_buffer(&from, 0, &to, 0, size);
        self.submit(encoder.finish());
        Some(to)
    }

    /// The context and parameter buffers for the execution being staged: the
    /// last ones, written again, while big enough; else new ones.
    pub(crate) fn uniform_buffers(
        &self,
        context: &[u8],
        params: &[u8],
    ) -> (wgpu::Buffer, wgpu::Buffer) {
        let mut held = self
            .uniforms
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let big_enough = held
            .as_ref()
            .is_some_and(|(context_buffer, params_buffer)| {
                context_buffer.size() >= context.len() as u64
                    && params_buffer.size() >= params.len().max(16) as u64
            });
        if !big_enough {
            // The smaller pair goes once the run is given to the device.
            if let Some((context_buffer, params_buffer)) = held.take() {
                self.spare.let_go(context_buffer);
                self.spare.let_go(params_buffer);
            }
            *held = Some((
                self.empty(
                    "context",
                    context.len().max(16),
                    wgpu::BufferUsages::UNIFORM,
                ),
                self.empty("params", params.len().max(16), wgpu::BufferUsages::UNIFORM),
            ));
        }
        let (context_buffer, params_buffer) = held.as_ref().expect("the buffers are held");
        let mut written = self
            .written
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // A new buffer holds nothing of what was written before it.
        if !big_enough {
            written.0.clear();
            written.1.clear();
        }
        if written.0 != context {
            self.queue.write_buffer(context_buffer, 0, context);
            written.0.clear();
            written.0.extend_from_slice(context);
        }
        if !params.is_empty() && written.1 != params {
            self.queue.write_buffer(params_buffer, 0, params);
            written.1.clear();
            written.1.extend_from_slice(params);
        }
        (context_buffer.clone(), params_buffer.clone())
    }

    /// Binds the inputs (memories as copies the steps may write), writes
    /// the context and the parameters, creates the output, and encodes
    /// the map pass over `domain` (sliced across `slice_limit` elements
    /// when a large map run is paced across submissions).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn stage(
        &self,
        pipeline: &Pipeline,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        domain: entelechy_abi::Domain,
        timing: Option<&Timing>,
        slice_limit: usize,
    ) -> Result<Staged, ExecutionError> {
        let element_count = domain.count();
        let can_slice = slice_limit < usize::MAX
            && timing.is_none()
            && module.manifest.mode == Mode::Map
            && module.manifest.memories.is_empty()
            && !pipeline.emits;
        let mut slices = if can_slice {
            wgsl::map_slices(domain, slice_limit)
        } else {
            let (gx, gy, gz, fold) = match module.manifest.mode {
                Mode::Group { size } => wgsl::group_dispatch(domain, size),
                _ => wgsl::dispatch(domain),
            };
            vec![(gx, gy, gz, fold, 0)]
        };
        let (groups_x, groups_y, groups_z, fold, slice_at) = slices.remove(0);
        let further_slices = slices;
        // Inputs: bind resident buffers directly; upload the rest once and
        // leave them resident.
        let mut input_buffers: Vec<wgpu::Buffer> = Vec::with_capacity(invocation.inputs.len());
        let mut input_extents: Vec<entelechy_abi::Domain> =
            Vec::with_capacity(invocation.inputs.len());
        // A memory's steps write it, so the bound buffer is a copy of the
        // input's, and that copy is the memory's output afterwards.
        let mut memory_outputs: Vec<(usize, wgpu::Buffer, entelechy_abi::Domain)> = Vec::new();
        let mut copies: Vec<(wgpu::Buffer, wgpu::Buffer, u64)> = Vec::new();
        for (index, handle) in invocation.inputs.iter().enumerate() {
            let extent = scope.size(*handle)?;
            input_extents.push(extent);
            let bound = self.bind_input(scope, *handle)?;
            let port = &module.manifest.inputs[index];
            if module.manifest.memories.contains(&port.name) {
                let size = (port.ty.schema.storage_words(extent.count()) * 4) as u64;
                let copy = self.spare.take(
                    &self.device,
                    size.max(16).div_ceil(16) * 16,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                );
                // Recorded below, into the same encoder as the pass and
                // before it, once the uniforms are settled.
                copies.push((bound, copy.clone(), size));
                memory_outputs.push((index, copy.clone(), extent));
                input_buffers.push(copy);
            } else {
                input_buffers.push(bound);
            }
        }
        // What the pass binds for its inputs: each its own, except packed ones,
        // copied into one buffer before it runs, each at the word offset the
        // parameters carry after their own.
        let mut bound_inputs: Vec<(u32, wgpu::Buffer)> =
            Vec::with_capacity(input_buffers.len() + 1);
        let mut packs: Vec<(wgpu::Buffer, u64, u64)> = Vec::new();
        let mut offsets: Vec<u32> = Vec::new();
        let mut packed_words = 0u64;
        for (index, buffer) in input_buffers.iter().enumerate() {
            if pipeline.packed.contains(&index) {
                let extent = input_extents[index];
                let words = module.manifest.inputs[index]
                    .ty
                    .schema
                    .storage_words(extent.count()) as u64;
                let offset = u32::try_from(packed_words).map_err(|_| {
                    ExecutionError::Device(
                        "the inputs packed into one buffer pass 2^32 words".to_owned(),
                    )
                })?;
                offsets.push(offset);
                packs.push((buffer.clone(), packed_words * 4, words * 4));
                packed_words += words;
            } else {
                bound_inputs.push((wgsl::INPUT_BINDING + index as u32, buffer.clone()));
            }
        }
        let pack = (!packs.is_empty()).then(|| {
            self.spare.take(
                &self.device,
                (packed_words * 4).max(16).div_ceil(16) * 16,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )
        });
        if let Some(pack) = &pack {
            bound_inputs.push((wgsl::PACKED_BINDING, pack.clone()));
        }
        // Context head (see `wgsl::CONTEXT_HEAD`), then one [w, h, d, 0] per
        // input; extents and the fold are the bits of `u32`s. Built after
        // `bind_input` so `write_words` is done with `self.staging`.
        let (context_buffer, params_buffer) = {
            let mut scratch = self
                .staging
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let StageScratch {
                context,
                params,
                words,
                ..
            } = &mut *scratch;
            context.clear();
            context.reserve((wgsl::CONTEXT_HEAD + input_extents.len() * 4) * 4);
            let push_u32 = |buf: &mut Vec<u8>, v: u32| buf.extend_from_slice(&v.to_le_bytes());
            let push_f32 = |buf: &mut Vec<u8>, v: f32| buf.extend_from_slice(&v.to_le_bytes());
            let input = &invocation.context.input;
            push_u32(context, domain.width());
            push_u32(context, domain.height());
            push_f32(context, invocation.context.float_time());
            push_f32(context, input.button_bits() as f32);
            push_f32(context, input.pointer[0]);
            push_f32(context, input.pointer[1]);
            push_u32(context, domain.depth());
            push_f32(context, invocation.context.rate);
            for &word in &input.key_words() {
                push_u32(context, word);
            }
            let window = invocation.context.window;
            push_f32(context, input.wheel[0]);
            push_f32(context, input.wheel[1]);
            push_f32(context, window[0]);
            push_f32(context, window[1]);
            // The third word is a one the shader compiler cannot see through
            // (`wgsl::OPAQUE_ONE`); the fourth is the slice offset.
            push_f32(context, invocation.context.scale);
            push_u32(context, fold);
            push_f32(context, 1.0);
            push_u32(context, slice_at);
            // The pad: two sticks in one vec4, the triggers and how many pads
            // are connected in the next, with the samples the sound device
            // holds, the bits of a `u32`.
            let sticks = input.sticks;
            push_f32(context, sticks[0][0]);
            push_f32(context, sticks[0][1]);
            push_f32(context, sticks[1][0]);
            push_f32(context, sticks[1][1]);
            push_f32(context, sticks[2][0]);
            push_f32(context, sticks[2][1]);
            push_f32(context, input.pads as f32);
            push_u32(context, invocation.context.queued.min(i32::MAX as u32));
            // The characters typed, four a slot.
            for &word in &input.typed_words() {
                push_u32(context, word);
            }
            for extent in &input_extents {
                push_u32(context, extent.width());
                push_u32(context, extent.height());
                push_u32(context, extent.depth());
                push_u32(context, 0);
            }
            fill_param_bytes(&invocation.params, words, params);
            if !offsets.is_empty() {
                // The offsets start at the first slot after the parameters'.
                params.truncate(wgsl::param_slots(module).1 * 16);
                for offset in &offsets {
                    params.extend_from_slice(&offset.to_le_bytes());
                }
                params.resize(params.len().div_ceil(16) * 16, 0);
            }
            self.uniform_buffers(context, params)
        };
        let output_schema = &module.manifest.outputs[0].ty.schema;
        let output_words = output_schema.value_words();
        // A scatter's map pass stores a pair per source element; the
        // rest store elements as the output keeps them.
        let map_bytes = if pipeline.emits {
            // A scatter that emits lands every pair from its entry, the
            // one it returns too: its pairs buffer is never written.
            16
        } else if matches!(module.manifest.mode, Mode::Scatter { .. }) {
            element_count * wgsl::type_words(module, module.entry.result_type()) * 4
        } else {
            output_schema.storage_words(element_count) * 4
        };
        let output_buffer = self.spare.take(
            &self.device,
            map_bytes as u64,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: wgsl::CONTEXT_BINDING,
                resource: context_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: wgsl::PARAMS_BINDING,
                resource: params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: wgsl::OUTPUT_BINDING,
                resource: output_buffer.as_entire_binding(),
            },
        ];
        for (binding, buffer) in &bound_inputs {
            entries.push(wgpu::BindGroupEntry {
                binding: *binding,
                resource: buffer.as_entire_binding(),
            });
        }
        // The failure word and its readback, from the pile, cleared below
        // before the pass. A whole sum records its first total that does not
        // fit there too.
        let failure = (pipeline.fails || wgsl::sums_whole(module)).then(|| self.failure_pair());
        if let (Some((failure, _)), true) = (&failure, pipeline.fails) {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::FAILURE_BINDING,
                resource: failure.as_entire_binding(),
            });
        }
        // A map's further outputs, each a buffer of its own words.
        let further_outputs: Vec<wgpu::Buffer> = module
            .manifest
            .outputs
            .iter()
            .take(wgsl::written_outputs(module))
            .skip(1)
            .map(|port| {
                self.spare.take(
                    &self.device,
                    (port.ty.schema.storage_words(element_count) * 4) as u64,
                    wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST,
                )
            })
            .collect();
        for (k, buffer) in further_outputs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::output_binding(module, k + 1),
                resource: buffer.as_entire_binding(),
            });
        }
        // A scatter that emits combines into its target as the entry
        // runs, so the target is made, and bound, here.
        let target = match module.manifest.mode {
            Mode::Scatter { base, .. } if pipeline.emits => {
                let count = scope.size(invocation.inputs[base as usize])?.count();
                Some((base, count, self.target_buffer(count, output_words)))
            }
            _ => None,
        };
        if let Some((_, _, buffer)) = &target {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::EMIT_BINDING,
                resource: buffer.as_entire_binding(),
            });
        }
        let highs = target
            .as_ref()
            .filter(|_| wgsl::highs_beside(module))
            .map(|(_, count, _)| self.target_buffer(*count, output_words));
        if let Some(highs) = &highs {
            entries.push(wgpu::BindGroupEntry {
                binding: wgsl::EMIT_HIGHS_BINDING,
                resource: highs.as_entire_binding(),
            });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy kernel"),
            layout: &pipeline.main.1,
            entries: &entries,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy"),
            });
        for (from, to, size) in &copies {
            encoder.copy_buffer_to_buffer(from, 0, to, 0, *size);
        }
        if let Some(pack) = &pack {
            for (from, at, size) in &packs {
                if *size > 0 {
                    encoder.copy_buffer_to_buffer(from, 0, pack, *at, *size);
                }
            }
        }
        if let (Some((base, count, buffer)), Some(scatter)) = (&target, &pipeline.scatter) {
            self.prepare_target(
                &mut encoder,
                scatter,
                buffer,
                highs.as_ref(),
                &input_buffers[*base as usize],
                &context_buffer,
                *count,
                output_words,
            );
        }
        // No failure yet, no work taken, and the most the run may take;
        // written before this submission's passes run.
        if let Some((failure, _)) = &failure {
            let limit = entelechy_rules::Run::new(scope.work()).limit();
            let mut record = [0u8; 32];
            record[8..12].copy_from_slice(&limit.to_le_bytes());
            self.queue.write_buffer(failure, 0, &record);
        }
        // A narrow output whose elements share words is stored by atomic
        // or, which needs every word it lands in to start at zero.
        for (k, buffer) in std::iter::once(&output_buffer)
            .chain(&further_outputs)
            .enumerate()
        {
            if k < wgsl::written_outputs(module) && wgsl::output_shares_words(module, k) {
                encoder.clear_buffer(buffer, 0, None);
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy kernel"),
                timestamp_writes: timing.map(|timing| timing.compute(Some(0), Some(1))),
            });
            pass.set_pipeline(&pipeline.main.0);
            pass.set_bind_group(0, &bind_group, &[]);
            // A scan walks its rows in one invocation.
            if module.manifest.mode == Mode::Scan {
                pass.dispatch_workgroups(1, 1, 1);
            } else {
                pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            }
        }
        // An emitting scatter of Longs by halves landed the high words: the
        // low ones are landed by the entry run again over the same inputs,
        // emitting the same pairs. Its work is counted on a record of its
        // own, never read, as the first run's counts the run's.
        if let (
            Some((base, count, buffer)),
            Some(Scatter::Parallel {
                halves:
                    Some(Halves {
                        lows,
                        entry: Some(entry),
                        ..
                    }),
                ..
            }),
        ) = (&target, &pipeline.scatter)
        {
            self.lows(
                &mut encoder,
                lows,
                &context_buffer,
                &input_buffers[*base as usize],
                buffer,
                count * output_words,
            );
            let counted = failure.as_ref().map(|_| {
                let counted = self.spare.take(
                    &self.device,
                    wgsl::FAILURE_BYTES,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                );
                let limit = entelechy_rules::Run::new(scope.work()).limit();
                let mut record = [0u8; 32];
                record[8..12].copy_from_slice(&limit.to_le_bytes());
                self.queue.write_buffer(&counted, 0, &record);
                counted
            });
            let entries: Vec<wgpu::BindGroupEntry> = entries
                .iter()
                .map(|entry| match (&counted, entry.binding) {
                    (Some(counted), wgsl::FAILURE_BINDING) => wgpu::BindGroupEntry {
                        binding: wgsl::FAILURE_BINDING,
                        resource: counted.as_entire_binding(),
                    },
                    _ => entry.clone(),
                })
                .collect();
            let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("entelechy kernel lows"),
                layout: &entry.1,
                entries: &entries,
            });
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy kernel lows"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&entry.0);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(groups_x, groups_y, groups_z);
            if let Some(counted) = counted {
                self.spare.keep(counted, None);
            }
        }
        Ok(Staged {
            encoder,
            bind_group,
            further_slices,
            context_buffer,
            params_buffer,
            input_buffers,
            bound_inputs,
            memory_outputs,
            output_buffer,
            further_outputs,
            output_words,
            element_count,
            failure,
            target: target.map(|(_, _, buffer)| buffer),
            highs,
        })
    }
}
