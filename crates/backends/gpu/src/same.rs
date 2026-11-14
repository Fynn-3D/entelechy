//! Whether two resident Fields hold the same words, compared on the
//! device: one pass sets a word where any two differ, and only that word
//! comes back.

use super::*;

const SAME_WGSL: &str = "
@group(0) @binding(0) var<storage, read> a: array<u32>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
@group(0) @binding(2) var<storage, read_write> differs: atomic<u32>;
@group(0) @binding(3) var<uniform> words: vec4<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let stride = groups.x * 256u;
    for (var i = id.x; i < words.x; i += stride) {
        if a[i] != b[i] {
            atomicStore(&differs, 1u);
            return;
        }
    }
}
";

/// The comparing pass and its scratch buffers, built once per backend and
/// used under [`GpuBackend::lock`].
pub(crate) struct SamePipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    differs: wgpu::Buffer,
    count: wgpu::Buffer,
    back: wgpu::Buffer,
}

pub(crate) fn build(device: &wgpu::Device, wait: Wait) -> Result<SamePipeline, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy same"),
        source: wgpu::ShaderSource::Wgsl(SAME_WGSL.into()),
    });
    let entry = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy same"),
        entries: &[
            entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
            entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
            entry(3, wgpu::BufferBindingType::Uniform),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy same"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy same"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = |label: &'static str, size: u64, usage: wgpu::BufferUsages| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let differs = buffer(
        "entelechy same",
        4,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let count = buffer(
        "entelechy same count",
        16,
        wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    );
    let back = buffer(
        "entelechy same back",
        4,
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    if let Some(error) = wait.block_on(guard.pop()) {
        return Err(format!("same pipeline: {}", device_error(&error)));
    }
    Ok(SamePipeline {
        pipeline,
        layout,
        differs,
        count,
        back,
    })
}

impl GpuBackend {
    /// Whether the first `byte_len` bytes of two device buffers are equal.
    pub(crate) fn same_buffers(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        byte_len: usize,
    ) -> Result<bool, String> {
        let same = self
            .same
            .get_or_init(|| build(&self.device, self.wait))
            .as_ref()
            .map_err(Clone::clone)?;
        let _guard = self
            .lock
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let words = u32::try_from(byte_len / 4).map_err(|_| "a Field too long to compare")?;
        let mut count_bytes = [0u8; 16];
        count_bytes[..4].copy_from_slice(&words.to_le_bytes());
        self.queue.write_buffer(&same.count, 0, &count_bytes);
        let group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy same"),
            layout: &same.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: same.differs.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: same.count.as_entire_binding(),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("entelechy same"),
            });
        encoder.clear_buffer(&same.differs, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("entelechy same"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&same.pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(words.div_ceil(256).clamp(1, 65535), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&same.differs, 0, &same.back, 0, 4);
        let submitted = self.submit(encoder.finish());
        self.spent
            .time(|| {
                self.wait
                    .map_read_after(&self.device, &self.queue, same.back.slice(..), &submitted)
            })
            .map_err(|error| crate::told(&self.heard, error))?;
        let word = {
            let view = same
                .back
                .slice(..)
                .get_mapped_range()
                .map_err(|error| error.to_string())?;
            u32::from_le_bytes([view[0], view[1], view[2], view[3]])
        };
        same.back.unmap();
        Ok(word == 0)
    }
}
