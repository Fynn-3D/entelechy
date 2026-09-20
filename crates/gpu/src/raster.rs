//! The GPU side of the native `rasterize` operation: the device's
//! rasterizer through a render pipeline, with the same contract as
//! [`orla_rules::raster`]. Two float render targets carry the surface
//! attributes, a depth target settles order, and a compute pass compacts
//! the padded copies into the tight word layout the Host expects, so the
//! result is a resident grid like any other output.

use std::sync::OnceLock;

use orla_abi::{Handle, Invocation, Scope};
use orla_rules::ExecutionError;
use orla_rules::raster::{SURFACE_WORDS, check_rasterize, surface_schema};

use crate::{DeviceImage, GpuBackend};

const RASTER_WGSL: &str = "
struct Frame { size: vec2<f32>, vertex_count: u32, pad: u32 };
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;

struct Varyings {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) id: u32,
};

@vertex
fn vs(@builtin(vertex_index) vi: u32) -> Varyings {
    let tri = vi / 3u;
    let corner = vi - tri * 3u;
    let raw = bitcast<i32>(triangles[tri * 3u + corner]);
    let v = u32(clamp(raw, 0i, i32(frame.vertex_count) - 1i)) * 9u;
    var out: Varyings;
    out.clip = vec4<f32>(
        bitcast<f32>(vertices[v]),
        bitcast<f32>(vertices[v + 1u]),
        bitcast<f32>(vertices[v + 2u]),
        bitcast<f32>(vertices[v + 3u]));
    out.normal = vec3<f32>(
        bitcast<f32>(vertices[v + 4u]),
        bitcast<f32>(vertices[v + 5u]),
        bitcast<f32>(vertices[v + 6u]));
    out.uv = vec2<f32>(bitcast<f32>(vertices[v + 7u]), bitcast<f32>(vertices[v + 8u]));
    out.id = tri;
    return out;
}

struct Targets {
    @location(0) a: vec4<f32>,
    @location(1) b: vec4<f32>,
};

@fragment
fn fs(in: Varyings) -> Targets {
    var out: Targets;
    out.a = vec4<f32>(in.clip.z, in.normal);
    // The id is kept one above its value so a cleared zero reads as -1.
    out.b = vec4<f32>(in.uv, f32(in.id + 1u), 0.0);
    return out;
}
";

const COMPACT_WGSL: &str = "
struct Frame { width: u32, height: u32, row_words: u32, pad: u32 };
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> a: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frame.width || gid.y >= frame.height) {
        return;
    }
    let src = gid.y * frame.row_words + gid.x * 4u;
    let dst = (gid.y * frame.width + gid.x) * 7u;
    out[dst] = a[src];
    out[dst + 1u] = a[src + 1u];
    out[dst + 2u] = a[src + 2u];
    out[dst + 3u] = a[src + 3u];
    out[dst + 4u] = b[src];
    out[dst + 5u] = b[src + 1u];
    out[dst + 6u] = bitcast<u32>(i32(bitcast<f32>(b[src + 2u])) - 1i);
}
";

/// The pipelines, built once per backend.
pub(crate) struct RasterPipelines {
    render: wgpu::RenderPipeline,
    render_layout: wgpu::BindGroupLayout,
    compact: wgpu::ComputePipeline,
    compact_layout: wgpu::BindGroupLayout,
}

const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

fn build(device: &wgpu::Device) -> Result<RasterPipelines, String> {
    let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let raster = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("orla rasterize"),
        source: wgpu::ShaderSource::Wgsl(RASTER_WGSL.into()),
    });
    let storage = |binding: u32, visibility: wgpu::ShaderStages, read_only: bool| {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    };
    let uniform = |binding: u32, visibility: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let render_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orla rasterize"),
        entries: &[
            uniform(0, wgpu::ShaderStages::VERTEX),
            storage(1, wgpu::ShaderStages::VERTEX, true),
            storage(2, wgpu::ShaderStages::VERTEX, true),
        ],
    });
    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("orla rasterize"),
        bind_group_layouts: &[Some(&render_layout)],
        immediate_size: 0,
    });
    let target = Some(wgpu::ColorTargetState {
        format: TARGET_FORMAT,
        blend: None,
        write_mask: wgpu::ColorWrites::ALL,
    });
    let render = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("orla rasterize"),
        layout: Some(&render_pipeline_layout),
        vertex: wgpu::VertexState {
            module: &raster,
            entry_point: Some("vs"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &raster,
            entry_point: Some("fs"),
            compilation_options: Default::default(),
            targets: &[target.clone(), target],
        }),
        multiview_mask: None,
        cache: None,
    });
    let compact_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("orla compact"),
        source: wgpu::ShaderSource::Wgsl(COMPACT_WGSL.into()),
    });
    let compact_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orla compact"),
        entries: &[
            uniform(0, wgpu::ShaderStages::COMPUTE),
            storage(1, wgpu::ShaderStages::COMPUTE, true),
            storage(2, wgpu::ShaderStages::COMPUTE, true),
            storage(3, wgpu::ShaderStages::COMPUTE, false),
        ],
    });
    let compact_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("orla compact"),
        bind_group_layouts: &[Some(&compact_layout)],
        immediate_size: 0,
    });
    let compact = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("orla compact"),
        layout: Some(&compact_pipeline_layout),
        module: &compact_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    if let Some(error) = pollster::block_on(guard.pop()) {
        return Err(format!("rasterize pipeline: {error}"));
    }
    Ok(RasterPipelines {
        render,
        render_layout,
        compact,
        compact_layout,
    })
}

fn target(
    device: &wgpu::Device,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

/// Rasterizes on the device into a new resident surface grid.
pub(crate) fn rasterize(
    gpu: &GpuBackend,
    pipelines: &OnceLock<Result<RasterPipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles) = check_rasterize(invocation, scope)?;
    let pipelines = match pipelines.get_or_init(|| build(&gpu.device)) {
        Ok(pipelines) => pipelines,
        Err(error) => return Err(ExecutionError::Device(error.clone())),
    };
    let _guard = gpu
        .lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let domain = invocation.domain;
    let (width, height) = (domain.width(), domain.height());
    let vertex_count = scope.size(vertices)?.count() as u32;
    let triangle_count = scope.size(triangles)?.count() as u32;
    let vertex_buffer = gpu.bind_input(scope, vertices, 0)?;
    let triangle_buffer = gpu.bind_input(scope, triangles, 1)?;

    let frame: [u32; 4] = [
        (width as f32).to_bits(),
        (height as f32).to_bits(),
        vertex_count,
        0,
    ];
    let frame_bytes: Vec<u8> = frame.iter().flat_map(|v| v.to_le_bytes()).collect();
    let frame_buffer = gpu.buffer("rasterize frame", &frame_bytes, wgpu::BufferUsages::UNIFORM);
    let render_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("orla rasterize"),
        layout: &pipelines.render_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: frame_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: triangle_buffer.as_entire_binding(),
            },
        ],
    });
    let a = target(&gpu.device, "surface a", width, height, TARGET_FORMAT);
    let b = target(&gpu.device, "surface b", width, height, TARGET_FORMAT);
    let depth = target(&gpu.device, "surface depth", width, height, DEPTH_FORMAT);
    let a_view = a.create_view(&Default::default());
    let b_view = b.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());

    // Copies of a texture need rows padded to 256 bytes.
    let row_bytes = (width as u64 * 16).div_ceil(256) * 256;
    let padded_size = row_bytes * height as u64;
    let padded = |label: &str| {
        gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    };
    let a_copy = padded("surface a copy");
    let b_copy = padded("surface b copy");
    let output_size = width as u64 * height as u64 * SURFACE_WORDS as u64 * 4;
    let output_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface"),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let compact_frame: [u32; 4] = [width, height, (row_bytes / 4) as u32, 0];
    let compact_bytes: Vec<u8> = compact_frame.iter().flat_map(|v| v.to_le_bytes()).collect();
    let compact_uniform = gpu.buffer("compact frame", &compact_bytes, wgpu::BufferUsages::UNIFORM);
    let compact_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("orla compact"),
        layout: &pipelines.compact_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: compact_uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: a_copy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: b_copy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("orla rasterize"),
        });
    {
        fn attachment(
            view: &wgpu::TextureView,
            clear: wgpu::Color,
        ) -> Option<wgpu::RenderPassColorAttachment<'_>> {
            Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
                    store: wgpu::StoreOp::Store,
                },
            })
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("orla rasterize"),
            color_attachments: &[
                attachment(
                    &a_view,
                    wgpu::Color {
                        r: 1.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    },
                ),
                attachment(&b_view, wgpu::Color::TRANSPARENT),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipelines.render);
        pass.set_bind_group(0, &render_group, &[]);
        pass.draw(0..triangle_count * 3, 0..1);
    }
    for (texture, copy) in [(&a, &a_copy), (&b, &b_copy)] {
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: copy,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes as u32),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("orla compact"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipelines.compact);
        pass.set_bind_group(0, &compact_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }
    let guard = gpu.device.push_error_scope(wgpu::ErrorFilter::Validation);
    gpu.queue.submit(Some(encoder.finish()));
    let _ = gpu.device.poll(wgpu::PollType::Poll);
    if let Some(error) = pollster::block_on(guard.pop()) {
        return Err(ExecutionError::Device(format!("rasterize: {error}")));
    }
    let output = scope.adopt_resident(
        surface_schema(),
        domain,
        Box::new(DeviceImage {
            buffer: output_buffer,
        }),
    )?;
    Ok(vec![output])
}
