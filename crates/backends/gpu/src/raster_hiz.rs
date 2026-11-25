//! Conservative MAX-Z Hierarchical-Z (Hi-Z) pyramid for GPU-driven 2-phase
//! occlusion culling in [`crate::raster`].
//!
//! Mip 0 has hardware dimensions `(max(1, W >> 1), max(1, H >> 1))` and each
//! subsequent mip halves via floor division (`max(1, w >> 1)`). Whenever a
//! parent dimension is odd, the last texel reduces 3 source texels along that
//! axis so every texel at mip `L` conservatively covers `[i * 2^(L+1),
//! (i+1) * 2^(L+1) - 1]` (or through `W - 1` / `H - 1` on the border).
//! Therefore any screen pixel bbox `[x0..=x1, y0..=y1]` is conservatively
//! bounded by at most a 2x2 texel block at mip `ceil_log2(max_span) - 1`.

use crate::WgpuBackend;

/// Minimum triangle count to trigger the 2-phase opaque Hi-Z pre-pass.
/// Below this threshold, single-pass visibility rasterization is cheaper than
/// encoding a depth pre-pass + mip pyramid.
pub(crate) const HIZ_TRI_THRESHOLD: u32 = 16_384;

/// Core WebGPU storage-writable single-channel depth format for the pyramid.
pub(crate) const HIZ_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Float;

const HIZ_WGSL: &str = "
struct HizLevel {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    has_before: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> lvl: HizLevel;
@group(0) @binding(1) var depth_tex: texture_depth_2d;
@group(0) @binding(2) var<storage, read> before: array<vec2<u32>>;
@group(0) @binding(3) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn mip0(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= lvl.dst_w || id.y >= lvl.dst_h {
        return;
    }
    let sx0 = id.x * 2u;
    let sy0 = id.y * 2u;
    let sx1 = select(min(sx0 + 1u, lvl.src_w - 1u), lvl.src_w - 1u, id.x + 1u == lvl.dst_w);
    let sy1 = select(min(sy0 + 1u, lvl.src_h - 1u), lvl.src_h - 1u, id.y + 1u == lvl.dst_h);
    var max_d = 0.0;
    for (var sy = sy0; sy <= sy1; sy++) {
        for (var sx = sx0; sx <= sx1; sx++) {
            var d = textureLoad(depth_tex, vec2<i32>(i32(sx), i32(sy)), 0);
            if lvl.has_before != 0u {
                let b = before[sy * lvl.src_w + sx];
                // Valid cover depth in [0.0, 1.0) has IEEE-754 bits strictly below 0x3F800000.
                if b.y < 0x3F800000u {
                    d = min(d, bitcast<f32>(b.y));
                }
            }
            max_d = max(max_d, d);
        }
    }
    textureStore(dst_tex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(max_d, 0.0, 0.0, 0.0));
}
";

const HIZ_DOWN_WGSL: &str = "
struct HizLevel {
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
    has_before: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> lvl: HizLevel;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var dst_tex: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8, 1)
fn down(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= lvl.dst_w || id.y >= lvl.dst_h {
        return;
    }
    let sx0 = id.x * 2u;
    let sy0 = id.y * 2u;
    let sx1 = select(min(sx0 + 1u, lvl.src_w - 1u), lvl.src_w - 1u, id.x + 1u == lvl.dst_w);
    let sy1 = select(min(sy0 + 1u, lvl.src_h - 1u), lvl.src_h - 1u, id.y + 1u == lvl.dst_h);
    var max_d = 0.0;
    for (var sy = sy0; sy <= sy1; sy++) {
        for (var sx = sx0; sx <= sx1; sx++) {
            let d = textureLoad(src_tex, vec2<i32>(i32(sx), i32(sy)), 0).x;
            max_d = max(max_d, d);
        }
    }
    textureStore(dst_tex, vec2<i32>(i32(id.x), i32(id.y)), vec4<f32>(max_d, 0.0, 0.0, 0.0));
}
";

/// Compute pipelines and a 1x1 placeholder texture for Hi-Z pyramid construction.
pub(crate) struct HizPipelines {
    mip0_layout: wgpu::BindGroupLayout,
    mip0: wgpu::ComputePipeline,
    down_layout: wgpu::BindGroupLayout,
    down: wgpu::ComputePipeline,
    /// 1x1 dummy texture bound when Hi-Z culling is inactive for a draw.
    pub(crate) empty_view: wgpu::TextureView,
    _empty_tex: wgpu::Texture,
}

pub(crate) fn build_hiz_pipelines(device: &wgpu::Device) -> HizPipelines {
    let c = wgpu::ShaderStages::COMPUTE;
    let uniform = wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: c,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage_tex = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: c,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HIZ_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    };
    let mip0_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy rasterize hiz mip0"),
        entries: &[
            uniform,
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: c,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: c,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage_tex(3),
        ],
    });
    let mip0_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy rasterize hiz mip0"),
        bind_group_layouts: &[Some(&mip0_layout)],
        immediate_size: 0,
    });
    let mip0_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy rasterize hiz mip0"),
        source: wgpu::ShaderSource::Wgsl(HIZ_WGSL.into()),
    });
    let mip0 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy rasterize hiz mip0"),
        layout: Some(&mip0_pl),
        module: &mip0_mod,
        entry_point: Some("mip0"),
        compilation_options: Default::default(),
        cache: None,
    });

    let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("entelechy rasterize hiz down"),
        entries: &[
            uniform,
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: c,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            storage_tex(2),
        ],
    });
    let down_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("entelechy rasterize hiz down"),
        bind_group_layouts: &[Some(&down_layout)],
        immediate_size: 0,
    });
    let down_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("entelechy rasterize hiz down"),
        source: wgpu::ShaderSource::Wgsl(HIZ_DOWN_WGSL.into()),
    });
    let down = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("entelechy rasterize hiz down"),
        layout: Some(&down_pl),
        module: &down_mod,
        entry_point: Some("down"),
        compilation_options: Default::default(),
        cache: None,
    });

    let empty_tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("entelechy hiz empty"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: HIZ_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let empty_view = empty_tex.create_view(&Default::default());

    HizPipelines {
        mip0_layout,
        mip0,
        down_layout,
        down,
        empty_view,
        _empty_tex: empty_tex,
    }
}

/// Returns `(w0, h0, mip_count)` for a canvas of size `(width, height)`.
pub(crate) fn hiz_dims(width: u32, height: u32) -> (u32, u32, u32) {
    let w0 = (width >> 1).max(1);
    let h0 = (height >> 1).max(1);
    let mip_count = w0.max(h0).ilog2() + 1;
    (w0, h0, mip_count)
}

/// Pre-encodes uniform payloads for every level of the Hi-Z pyramid so they
/// can be allocated atomically from the batch uniform slab before encoding.
pub(crate) fn hiz_level_payloads(width: u32, height: u32, has_before: bool) -> Vec<[u8; 32]> {
    let (w0, h0, mip_count) = hiz_dims(width, height);
    let mut out = Vec::with_capacity(mip_count as usize);
    let mut src_w = width;
    let mut src_h = height;
    let mut dst_w = w0;
    let mut dst_h = h0;
    for lvl in 0..mip_count {
        let words: [u32; 8] = [
            src_w,
            src_h,
            dst_w,
            dst_h,
            u32::from(lvl == 0 && has_before),
            0,
            0,
            0,
        ];
        let mut bytes = [0u8; 32];
        for (slot, word) in bytes.chunks_exact_mut(4).zip(words) {
            slot.copy_from_slice(&word.to_le_bytes());
        }
        out.push(bytes);
        src_w = dst_w;
        src_h = dst_h;
        dst_w = (dst_w >> 1).max(1);
        dst_h = (dst_h >> 1).max(1);
    }
    out
}

/// Acquires a pooled mipmapped `R32Float` Hi-Z texture for `(width, height)`.
pub(crate) fn take_hiz_texture(gpu: &WgpuBackend, width: u32, height: u32) -> wgpu::Texture {
    let (w0, h0, mip_count) = hiz_dims(width, height);
    gpu.targets().take_mipmapped(
        &gpu.device,
        "surface hiz",
        w0,
        h0,
        mip_count,
        HIZ_FORMAT,
        wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
    )
}

/// Encodes all mip levels of the conservative MAX-Z pyramid into `encoder`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_hiz_pyramid(
    device: &wgpu::Device,
    pipelines: &HizPipelines,
    encoder: &mut wgpu::CommandEncoder,
    depth_view: &wgpu::TextureView,
    before_buf: &wgpu::Buffer,
    hiz_tex: &wgpu::Texture,
    width: u32,
    height: u32,
    level_uniforms: &[crate::batch::UniformSlice],
) {
    let (w0, h0, mip_count) = hiz_dims(width, height);
    let mip_views: Vec<wgpu::TextureView> = (0..mip_count)
        .map(|lvl| {
            hiz_tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some("hiz mip"),
                format: Some(HIZ_FORMAT),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: lvl,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: None,
            })
        })
        .collect();

    let mip0_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("entelechy rasterize hiz mip0"),
        layout: &pipelines.mip0_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: level_uniforms[0].binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: before_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::TextureView(&mip_views[0]),
            },
        ],
    });
    let mut down_bgs = Vec::with_capacity((mip_count as usize).saturating_sub(1));
    for lvl in 1..mip_count as usize {
        down_bgs.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy rasterize hiz down"),
            layout: &pipelines.down_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: level_uniforms[lvl].binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&mip_views[lvl - 1]),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&mip_views[lvl]),
                },
            ],
        }));
    }

    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("entelechy rasterize hiz"),
        timestamp_writes: None,
    });
    pass.set_pipeline(&pipelines.mip0);
    pass.set_bind_group(0, &mip0_bg, &[]);
    pass.dispatch_workgroups(w0.div_ceil(8), h0.div_ceil(8), 1);

    let mut dst_w = (w0 >> 1).max(1);
    let mut dst_h = (h0 >> 1).max(1);
    if !down_bgs.is_empty() {
        pass.set_pipeline(&pipelines.down);
        for bg in &down_bgs {
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(dst_w.div_ceil(8), dst_h.div_ceil(8), 1);
            dst_w = (dst_w >> 1).max(1);
            dst_h = (dst_h >> 1).max(1);
        }
    }
}
