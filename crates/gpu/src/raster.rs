//! The GPU side of the native `rasterize` operation: the device's
//! rasterizer through a render pipeline, with the same contract as
//! [`orla_rules::raster`]. Two float render targets carry the surface
//! attributes, a depth target settles order, and a compute pass compacts
//! the targets into the tight word layout the Host expects, reading
//! them where they were written, so the result is a resident grid like
//! any other output and the surface is gathered once.

use std::sync::OnceLock;

use orla_abi::{Handle, Invocation, Scope};
use orla_rules::ExecutionError;
use orla_rules::raster::{SURFACE_WORDS, check_rasterize, surface_schema};

use crate::{DeviceImage, GpuBackend};

const RASTER_WGSL: &str = "
struct Frame {
    size: vec2<f32>,
    vertex_count: u32,
    mask_width: u32,
    mask_height: u32,
    cutout: f32,
    masked: u32,
    pad: u32,
};
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> triangles: array<u32>;
@group(0) @binding(3) var<storage, read> mask: array<u32>;

// The mask's alpha where `at_linear` would read it: the four texels
// around the place, blended, clamped at the edges. The reference
// rasterizer reads it the same way, so a cutout's edge is the same
// pixel on either.
fn mask_alpha(uv: vec2<f32>) -> f32 {
    let w = i32(frame.mask_width);
    let h = i32(frame.mask_height);
    let sx = uv.x * f32(w) - 0.5;
    let sy = uv.y * f32(h) - 0.5;
    let x0 = i32(floor(sx));
    let y0 = i32(floor(sy));
    let fx = sx - floor(sx);
    let fy = sy - floor(sy);
    let at = fn_at(x0, y0, w, h);
    let bx = fn_at(x0 + 1, y0, w, h);
    let ay = fn_at(x0, y0 + 1, w, h);
    let by = fn_at(x0 + 1, y0 + 1, w, h);
    let top = at + (bx - at) * fx;
    let bottom = ay + (by - ay) * fx;
    return top + (bottom - top) * fy;
}

fn fn_at(x: i32, y: i32, w: i32, h: i32) -> f32 {
    let cx = clamp(x, 0, max(w - 1, 0));
    let cy = clamp(y, 0, max(h - 1, 0));
    return bitcast<f32>(mask[u32(cy * w + cx) * 4u + 3u]);
}

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
    if (frame.masked != 0u && mask_alpha(in.uv) < frame.cutout) {
        discard;
    }
    var out: Targets;
    out.a = vec4<f32>(in.clip.z, in.normal);
    // The id is kept one above its value so a cleared zero reads as -1.
    out.b = vec4<f32>(in.uv, f32(in.id + 1u), 0.0);
    return out;
}
";

const COMPACT_WGSL: &str = "
struct Frame { width: u32, height: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var a: texture_2d<f32>;
@group(0) @binding(2) var b: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<u32>;

// The targets are read where they were written, so the surface is
// gathered once rather than copied out and gathered.
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= frame.width || gid.y >= frame.height) {
        return;
    }
    let at = vec2<i32>(i32(gid.x), i32(gid.y));
    let first = textureLoad(a, at, 0);
    let second = textureLoad(b, at, 0);
    let dst = (gid.y * frame.width + gid.x) * 7u;
    out[dst] = bitcast<u32>(first.x);
    out[dst + 1u] = bitcast<u32>(first.y);
    out[dst + 2u] = bitcast<u32>(first.z);
    out[dst + 3u] = bitcast<u32>(first.w);
    out[dst + 4u] = bitcast<u32>(second.x);
    out[dst + 5u] = bitcast<u32>(second.y);
    // The id was kept one above its value so a cleared zero reads as -1.
    out[dst + 6u] = bitcast<u32>(i32(second.z) - 1i);
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
            uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT),
            storage(1, wgpu::ShaderStages::VERTEX, true),
            storage(2, wgpu::ShaderStages::VERTEX, true),
            storage(3, wgpu::ShaderStages::FRAGMENT, true),
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
    let sampled = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let compact_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("orla compact"),
        entries: &[
            uniform(0, wgpu::ShaderStages::COMPUTE),
            sampled(1),
            sampled(2),
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

/// A target of this shape, from the backend's pile. Written as an
/// attachment and read by the compact pass where it was written,
/// rather than copied out first.
fn target(
    gpu: &GpuBackend,
    label: &str,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    gpu.targets().take(
        &gpu.device,
        label,
        width,
        height,
        format,
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
    )
}

/// Rasterizes on the device into a new resident surface grid.
pub(crate) fn rasterize(
    gpu: &GpuBackend,
    pipelines: &OnceLock<Result<RasterPipelines, String>>,
    invocation: &Invocation,
    scope: &mut Scope<'_>,
    cutout: bool,
) -> Result<Vec<Handle>, ExecutionError> {
    let (vertices, triangles, mask, alpha) = if cutout {
        let (vertices, triangles, mask, alpha) =
            orla_rules::raster::check_rasterize_cutout(invocation, scope)?;
        (vertices, triangles, Some(mask), alpha)
    } else {
        let (vertices, triangles) = check_rasterize(invocation, scope)?;
        (vertices, triangles, None, 0.0)
    };
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

    // The mask is always bound: without one it is a single white texel
    // the shader is told not to read.
    let mask_size = match mask {
        Some(mask) => scope.size(mask)?,
        None => orla_abi::Domain::Grid(1, 1),
    };
    let mask_buffer = match mask {
        Some(mask) => gpu.bind_input(scope, mask, 2)?,
        None => gpu.buffer(
            "rasterize mask",
            &[0u8; 16],
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        ),
    };
    let frame: [u32; 8] = [
        (width as f32).to_bits(),
        (height as f32).to_bits(),
        vertex_count,
        mask_size.width(),
        mask_size.height(),
        alpha.to_bits(),
        u32::from(mask.is_some()),
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
            wgpu::BindGroupEntry {
                binding: 3,
                resource: mask_buffer.as_entire_binding(),
            },
        ],
    });
    let a = target(gpu, "surface a", width, height, TARGET_FORMAT);
    let b = target(gpu, "surface b", width, height, TARGET_FORMAT);
    let depth = target(gpu, "surface depth", width, height, DEPTH_FORMAT);
    let a_view = a.create_view(&Default::default());
    let b_view = b.create_view(&Default::default());
    let depth_view = depth.create_view(&Default::default());

    let output_size = width as u64 * height as u64 * SURFACE_WORDS as u64 * 4;
    let output_buffer = gpu.spare().take(
        &gpu.device,
        output_size,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
    );
    let compact_frame: [u32; 4] = [width, height, 0, 0];
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
                resource: wgpu::BindingResource::TextureView(&a_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&b_view),
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
            spare: Some(gpu.spare()),
        }),
    )?;
    // The targets were this call's own; the pile takes them back for
    // the next rasterize of the same shape.
    for texture in [a, b, depth] {
        gpu.targets().keep(texture);
    }
    Ok(vec![output])
}
