//! What only a page's device does: a pipeline is asked of the browser
//! first, to be compiled off the thread that answers the device.
//!
//! A browser compiles a pipeline made the plain way on that thread, the
//! first time it runs; a large one holds every page's device that long,
//! and a browser that watches the thread takes it for hung. One asked for
//! asynchronously is compiled beside it; the browser keeps pipelines by
//! what they are made of, so the plain one made after it is the same
//! pipeline, already compiled.

use js_sys::{Array, Function, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::wait::Wait;

thread_local! {
    /// Pipelines asked of the browser ahead ([`ask_ahead`]) and not yet
    /// waited for, keyed by `(device_id, shader_and_layout_hash)`.
    static ASKED: std::cell::RefCell<std::collections::HashMap<(u32, u64), js_sys::Promise>> =
        std::cell::RefCell::default();
    static NEXT_DEVICE_ID: std::cell::Cell<u32> = const { std::cell::Cell::new(1) };
    static NOOP_CATCH: Function = Function::new_no_args("");
}

/// Stable numeric tag attached to each JS `GPUDevice` so promises from a
/// closed or lost device never collide with a new session's device.
fn device_tag(device: &JsValue) -> u32 {
    let key = JsValue::from_str("__ent_dev_id");
    if let Ok(existing) = Reflect::get(device, &key)
        && let Some(id) = existing.as_f64()
    {
        return id as u32;
    }
    let id = NEXT_DEVICE_ID.with(|cell| {
        let cur = cell.get();
        cell.set(cur.wrapping_add(1).max(1));
        cur
    });
    let _ = Reflect::set(device, &key, &JsValue::from_f64(f64::from(id)));
    id
}

/// What a pipeline is made of, as [`ASKED`] keeps it.
fn made_of(source: &str, entries: &[wgpu::BindGroupLayoutEntry]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    source.hash(&mut hasher);
    entries.hash(&mut hasher);
    hasher.finish()
}

fn describe_js_error(err: &JsValue) -> String {
    if let Some(s) = err.as_string() {
        return s;
    }
    if let Ok(msg) = Reflect::get(err, &"message".into())
        && let Some(s) = msg.as_string()
        && !s.is_empty()
    {
        return s;
    }
    format!("{err:?}")
}

/// Asks the browser to start compiling a pipeline as a Kernel is
/// prepared, and goes on: the browser compiles the pipelines a scene asks
/// for side by side, and [`compile_beside`] waits for this one when it is
/// first built.
pub(crate) fn ask_ahead(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) {
    let Some(device) = device.as_webgpu() else {
        return;
    };
    let device: &JsValue = device.as_ref();
    let key = (device_tag(device), made_of(source, entries));
    if ASKED.with(|asked_ahead| asked_ahead.borrow().contains_key(&key)) {
        return;
    }
    if let Ok(asked) = ask(device, label, source, entries) {
        ASKED.with(|asked_ahead| {
            asked_ahead.borrow_mut().insert(key, asked);
        });
    }
}

/// Why asynchronous WebGPU pipeline compilation in a page failed.
pub(crate) enum CompileError {
    /// The JS layout-descriptor bridge could not encode an entry before asking
    /// the browser (an engine bug that must surface instead of silently
    /// degrading to single-threaded CPU fallback).
    Bridge(String),
    /// The browser or WebGPU device refused or lost the pipeline.
    Device(String),
}

impl CompileError {
    fn message(&self) -> &str {
        match self {
            Self::Bridge(msg) | Self::Device(msg) => msg,
        }
    }
}

/// Asks the browser to compile the pipeline of `source`'s `main` with one
/// group of `entries` (or takes what [`ask_ahead`] asked), and waits until
/// it has, parked. Returns `Err` immediately when the browser or device
/// refuses or loses the pipeline so dead-device builds fail fast.
pub(crate) fn compile_beside(
    device: &wgpu::Device,
    wait: Wait,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<(), CompileError> {
    let Some(device) = device.as_webgpu() else {
        return Ok(());
    };
    let device: &JsValue = device.as_ref();
    let traced = traced();
    if traced {
        tell(&format!(
            "compiling {label} ({} KiB of WGSL)",
            source.len().div_ceil(1024)
        ));
    }
    let started = entelechy_rules::clock::now();
    let key = (device_tag(device), made_of(source, entries));
    let ahead = ASKED.with(|asked_ahead| asked_ahead.borrow_mut().remove(&key));
    let asked = match ahead {
        Some(asked) => asked,
        None => match ask(device, label, source, entries) {
            Ok(asked) => asked,
            Err(error) => {
                if traced {
                    let took = entelechy_rules::clock::since(started).as_secs_f64() * 1000.0;
                    tell(&format!(
                        "refused {label} in {took:.0} ms ({})",
                        error.message()
                    ));
                }
                return Err(error);
            }
        },
    };
    let answer = wait.block_on(wasm_bindgen_futures::JsFuture::from(asked));
    let result = answer
        .map(|_| ())
        .map_err(|e| CompileError::Device(describe_js_error(&e)));
    if traced {
        let took = entelechy_rules::clock::since(started).as_secs_f64() * 1000.0;
        match &result {
            Ok(()) => tell(&format!("compiled {label} in {took:.0} ms")),
            Err(error) => tell(&format!(
                "refused {label} in {took:.0} ms ({})",
                error.message()
            )),
        }
    }
    result
}

/// Whether the page asked to be told each pipeline (`entelechy_trace`).
fn traced() -> bool {
    Reflect::get(&js_sys::global(), &"entelechy_trace".into()).is_ok_and(|on| on.is_truthy())
}

/// A line on the console, named Entelechy's.
fn tell(line: &str) {
    let Ok(console) = Reflect::get(&js_sys::global(), &"console".into()) else {
        return;
    };
    let _ = call(&console, "log", &format!("entelechy: {line}").into());
}

/// The browser's promise of the pipeline, marked handled so rejected warmups
/// on a lost device never fire unhandled-rejection console noise before
/// [`compile_beside`] awaits them.
fn ask(
    device: &JsValue,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<js_sys::Promise, CompileError> {
    let bindings = Array::new();
    for entry in entries {
        bindings.push(&binding(entry)?);
    }
    ask_device(device, label, source, bindings)
        .map_err(|e| CompileError::Device(describe_js_error(&e)))
}

fn ask_device(
    device: &JsValue,
    label: &str,
    source: &str,
    bindings: Array,
) -> Result<js_sys::Promise, JsValue> {
    let module = call(
        device,
        "createShaderModule",
        &object(&[("label", label.into()), ("code", source.into())])?,
    )?;
    let group = call(
        device,
        "createBindGroupLayout",
        &object(&[("label", label.into()), ("entries", bindings.into())])?,
    )?;
    let layout = call(
        device,
        "createPipelineLayout",
        &object(&[
            ("label", label.into()),
            ("bindGroupLayouts", Array::of1(&group).into()),
        ])?,
    )?;
    let compute = object(&[("module", module), ("entryPoint", "main".into())])?;
    let descriptor = object(&[
        ("label", label.into()),
        ("layout", layout),
        ("compute", compute),
    ])?;
    let promise: js_sys::Promise =
        call(device, "createComputePipelineAsync", &descriptor)?.dyn_into()?;
    NOOP_CATCH.with(|noop| {
        let _ = call(promise.as_ref(), "catch", noop.as_ref());
    });
    Ok(promise)
}

/// A bind group layout entry as WebGPU's `GPUBindGroupLayoutEntry` takes it:
/// uniform/storage buffers, sampled hardware textures, and hardware samplers.
fn binding(entry: &wgpu::BindGroupLayoutEntry) -> Result<JsValue, CompileError> {
    let js_err = |e: JsValue| CompileError::Bridge(describe_js_error(&e));
    let (field, layout) = match entry.ty {
        wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset,
            min_binding_size,
        } => {
            let kind = match ty {
                wgpu::BufferBindingType::Uniform => "uniform",
                wgpu::BufferBindingType::Storage { read_only: true } => "read-only-storage",
                wgpu::BufferBindingType::Storage { read_only: false } => "storage",
            };
            let buffer = object(&[
                ("type", kind.into()),
                ("hasDynamicOffset", has_dynamic_offset.into()),
                (
                    "minBindingSize",
                    min_binding_size
                        .map_or(0.0, |size| size.get() as f64)
                        .into(),
                ),
            ])
            .map_err(js_err)?;
            ("buffer", buffer)
        }
        wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled,
        } => {
            let sample = match sample_type {
                wgpu::TextureSampleType::Float { filterable: true } => "float",
                wgpu::TextureSampleType::Float { filterable: false } => "unfilterable-float",
                wgpu::TextureSampleType::Depth => "depth",
                wgpu::TextureSampleType::Sint => "sint",
                wgpu::TextureSampleType::Uint => "uint",
            };
            let dimension = match view_dimension {
                wgpu::TextureViewDimension::D1 => "1d",
                wgpu::TextureViewDimension::D2 => "2d",
                wgpu::TextureViewDimension::D2Array => "2d-array",
                wgpu::TextureViewDimension::Cube => "cube",
                wgpu::TextureViewDimension::CubeArray => "cube-array",
                wgpu::TextureViewDimension::D3 => "3d",
            };
            let texture = object(&[
                ("sampleType", sample.into()),
                ("viewDimension", dimension.into()),
                ("multisampled", multisampled.into()),
            ])
            .map_err(js_err)?;
            ("texture", texture)
        }
        wgpu::BindingType::Sampler(ty) => {
            let kind = match ty {
                wgpu::SamplerBindingType::Filtering => "filtering",
                wgpu::SamplerBindingType::NonFiltering => "non-filtering",
                wgpu::SamplerBindingType::Comparison => "comparison",
            };
            let sampler = object(&[("type", kind.into())]).map_err(js_err)?;
            ("sampler", sampler)
        }
        other => {
            return Err(CompileError::Bridge(format!(
                "unsupported WebGPU bind group layout entry: {other:?}"
            )));
        }
    };
    object(&[
        ("binding", entry.binding.into()),
        ("visibility", entry.visibility.bits().into()),
        (field, layout),
    ])
    .map_err(js_err)
}

fn object(fields: &[(&str, JsValue)]) -> Result<JsValue, JsValue> {
    let object = Object::new();
    for (name, value) in fields {
        Reflect::set(&object, &(*name).into(), value)?;
    }
    Ok(object.into())
}

fn call(target: &JsValue, method: &str, argument: &JsValue) -> Result<JsValue, JsValue> {
    let function: js_sys::Function = Reflect::get(target, &method.into())?.dyn_into()?;
    function.call1(target, argument)
}
