//! Presenting a resident image to a window without reading it back.
//!
//! The Player takes a [`Display`] from the backend before the backend goes
//! to a runtime, and hands it a window. The display draws the output image
//! with one fullscreen triangle sampling its storage buffer, letterboxed
//! and nearest-neighbor scaled, with an optional `0x00RRGGBB` overlay in
//! the window's top-left corner.

use std::any::Any;

use crate::{DeviceImage, GpuBackend};

/// The GPU device as a window reaches it: taken from a backend before the
/// backend goes to a runtime, sharing its device, so an interface draws
/// what the backend left resident; the runtime knows only a `Backend`.
#[derive(Clone)]
pub struct Display {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl GpuBackend {
    /// A display on this backend's device, for presenting what it leaves
    /// resident.
    pub fn display(&self) -> Display {
        let (instance, adapter, device, queue) = self.parts();
        Display {
            instance: instance.clone(),
            adapter: adapter.clone(),
            device: device.clone(),
            queue: queue.clone(),
        }
    }
}

/// Where a picture lands in a window: one rule for the device's presenter
/// and a CPU blit alike.
pub use entelechy_rules::placement::{Placement, placement};

/// The WGSL that blits a `vec4<f32>` storage buffer onto the surface.
pub const BLIT: &str = "
struct Frame {
    // window width, window height, image width, image height
    sizes: vec4<f32>,
    // placement x, y, width, height
    place: vec4<f32>,
    // 1 when the surface encodes sRGB itself (values are written linear);
    // then the overlay's width and height, zero for none
    flags: vec4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> image: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> overlay: array<u32>;

struct Vertex { @builtin(position) position: vec4<f32> };

@vertex
fn vs(@builtin(vertex_index) index: u32) -> Vertex {
    // One triangle covering the clip space.
    var corners = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return Vertex(vec4<f32>(corners[index], 0.0, 1.0));
}

fn encode(c: vec3<f32>) -> vec4<f32> {
    if (frame.flags.x > 0.5) {
        return vec4<f32>(pow(c, vec3<f32>(2.2)), 1.0);
    }
    return vec4<f32>(c, 1.0);
}

@fragment
fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    if (position.x < frame.flags.y && position.y < frame.flags.z) {
        let o = overlay[u32(position.y) * u32(frame.flags.y) + u32(position.x)];
        let c = vec3<f32>(f32((o >> 16u) & 255u), f32((o >> 8u) & 255u), f32(o & 255u)) / 255.0;
        return encode(c);
    }
    let p = position.xy - frame.place.xy;
    if (p.x < 0.0 || p.y < 0.0 || p.x >= frame.place.z || p.y >= frame.place.w) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let w = u32(frame.sizes.z);
    let h = u32(frame.sizes.w);
    let x = min(u32(p.x * frame.sizes.z / frame.place.z), w - 1u);
    let y = min(u32(p.y * frame.sizes.w / frame.place.w), h - 1u);
    let c = clamp(image[y * w + x].rgb, vec3<f32>(0.0), vec3<f32>(1.0));
    return encode(c);
}
";

/// Pixels drawn over the window's top-left corner: `0x00RRGGBB`, `width`
/// per row.
#[derive(Clone, Copy, Debug)]
pub struct Overlay<'a> {
    /// The pixels, row by row.
    pub pixels: &'a [u32],
    /// Width in window pixels.
    pub width: u32,
    /// Height in window pixels.
    pub height: u32,
}

/// A window surface a [`Display`] draws resident images into.
pub struct Presenter {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    srgb: bool,
}

impl Display {
    /// Configures a surface on `target` (a window handle the backend may
    /// keep) of the given size.
    pub fn presenter(
        &self,
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
    ) -> Result<Presenter, String> {
        let (instance, adapter, device) = (&self.instance, &self.adapter, &self.device);
        let surface = instance
            .create_surface(target)
            .map_err(|error| error.to_string())?;
        let capabilities = surface.get_capabilities(adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|format| !format.is_srgb())
            .or_else(|| capabilities.formats.first().copied())
            .ok_or_else(|| "the surface offers no formats".to_string())?;
        let mut config = surface
            .get_default_config(adapter, width.max(1), height.max(1))
            .ok_or_else(|| "the adapter cannot present to this surface".to_string())?;
        config.format = format;
        surface.configure(device, &config);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entelechy blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("entelechy blit"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("entelechy blit"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("entelechy blit"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        Ok(Presenter {
            surface,
            config,
            pipeline,
            layout,
            srgb: format.is_srgb(),
        })
    }

    /// Draws a resident image (`device`, the Host's resident object) into the
    /// surface, with `overlay` over its top-left corner. Returns where the
    /// image was placed, or `None` when the window takes no frame now
    /// (occluded, or the surface timed out). Nothing is read back.
    #[allow(clippy::too_many_arguments)]
    pub fn present(
        &self,
        presenter: &mut Presenter,
        device: &(dyn Any + Send + Sync),
        width: u32,
        height: u32,
        window_width: u32,
        window_height: u32,
        overlay: Option<Overlay<'_>>,
    ) -> Result<Option<Placement>, String> {
        let image = device
            .downcast_ref::<DeviceImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        let (gpu_device, queue) = (&self.device, &self.queue);
        if (presenter.config.width, presenter.config.height) != (window_width, window_height)
            && window_width > 0
            && window_height > 0
        {
            presenter.config.width = window_width;
            presenter.config.height = window_height;
            presenter.surface.configure(gpu_device, &presenter.config);
        }
        let placement = placement(width, height, window_width, window_height);
        let frame: [f32; 12] = [
            window_width as f32,
            window_height as f32,
            width as f32,
            height as f32,
            placement.x as f32,
            placement.y as f32,
            placement.width as f32,
            placement.height as f32,
            presenter.srgb as u8 as f32,
            overlay.map_or(0.0, |o| o.width as f32),
            overlay.map_or(0.0, |o| o.height as f32),
            0.0,
        ];
        let texture = match presenter.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(None);
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                presenter.surface.configure(gpu_device, &presenter.config);
                return Ok(None);
            }
            other => return Err(format!("no surface texture: {other:?}")),
        };
        // Made once the frame has a texture, so each is destroyed.
        let bytes: Vec<u8> = frame.iter().flat_map(|v| v.to_le_bytes()).collect();
        let uniform = self.buffer("blit frame", &bytes, wgpu::BufferUsages::UNIFORM);
        let overlay_bytes: Vec<u8> = overlay
            .map(|o| o.pixels.iter().flat_map(|p| p.to_le_bytes()).collect())
            .unwrap_or_else(|| vec![0; 4]);
        let overlay_buffer =
            self.buffer("blit overlay", &overlay_bytes, wgpu::BufferUsages::STORAGE);
        let bind_group = gpu_device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("entelechy blit"),
            layout: &presenter.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: overlay_buffer.as_entire_binding(),
                },
            ],
        });
        let view = texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = gpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("entelechy present"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("entelechy blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&presenter.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        // Read by this frame's blit alone.
        crate::wait::discard(uniform);
        crate::wait::discard(overlay_buffer);
        queue.present(texture);
        Ok(Some(placement))
    }

    /// A buffer of `contents`, as the backend makes one.
    fn buffer(&self, label: &str, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let size = (contents.len().max(16) as u64).div_ceil(16) * 16;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !contents.is_empty() {
            self.queue.write_buffer(&buffer, 0, contents);
        }
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_fits_and_centers() {
        assert_eq!(
            placement(2, 1, 8, 2),
            Placement {
                x: 2,
                y: 0,
                width: 4,
                height: 2
            }
        );
        assert_eq!(
            placement(2, 1, 4, 4),
            Placement {
                x: 0,
                y: 1,
                width: 4,
                height: 2
            }
        );
        assert_eq!(placement(0, 1, 4, 4), Placement::default());
    }

    #[test]
    fn the_blit_shader_compiles() {
        let Ok(gpu) = GpuBackend::new() else {
            eprintln!("skipping: no gpu");
            return;
        };
        let (_, _, device, _) = gpu.parts();
        let guard = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let _ = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("entelechy blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT.into()),
        });
        assert!(pollster::block_on(guard.pop()).is_none());
    }
}
