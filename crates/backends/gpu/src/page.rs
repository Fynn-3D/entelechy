//! What only a page's device does: a pipeline is asked of the browser
//! first, to be compiled off the thread that answers the device.
//!
//! A browser compiles a pipeline made the plain way on that thread, the
//! first time it runs; a large one holds every page's device that long,
//! and a browser that watches the thread takes it for hung. One asked for
//! asynchronously is compiled beside it; the browser keeps pipelines by
//! what they are made of, so the plain one made after it is the same
//! pipeline, already compiled.

use js_sys::{Array, Object, Reflect};
use wasm_bindgen::{JsCast, JsValue};

use crate::wait::Wait;

thread_local! {
    /// Pipelines asked of the browser ahead ([`ask_ahead`]) and not yet
    /// waited for, by what they are made of.
    static ASKED: std::cell::RefCell<std::collections::HashMap<u64, js_sys::Promise>> =
        std::cell::RefCell::default();
}

/// What a pipeline is made of, as [`ASKED`] keeps it.
fn made_of(label: &str, source: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    label.hash(&mut hasher);
    source.hash(&mut hasher);
    hasher.finish()
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
    if let Ok(asked) = ask(device, label, source, entries) {
        ASKED.with(|asked_ahead| {
            asked_ahead
                .borrow_mut()
                .insert(made_of(label, source), asked)
        });
    }
}

/// Asks the browser to compile the pipeline of `source`'s `main` with one
/// group of `entries` (or takes what [`ask_ahead`] asked), and waits until
/// it has, parked. Whatever the browser says against it is said again,
/// and caught, when the pipeline is made.
pub(crate) fn compile_beside(
    device: &wgpu::Device,
    wait: Wait,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) {
    let Some(device) = device.as_webgpu() else {
        return;
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
    let ahead = ASKED.with(|asked_ahead| asked_ahead.borrow_mut().remove(&made_of(label, source)));
    let asked = match ahead {
        Some(asked) => asked,
        None => match ask(device, label, source, entries) {
            Ok(asked) => asked,
            Err(_) => return,
        },
    };
    let answer = wait.block_on(wasm_bindgen_futures::JsFuture::from(asked));
    if traced {
        let took = entelechy_rules::clock::since(started).as_secs_f64() * 1000.0;
        let how = if answer.is_ok() {
            "compiled"
        } else {
            "refused"
        };
        tell(&format!("{how} {label} in {took:.0} ms"));
    }
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

/// The browser's promise of the pipeline.
fn ask(
    device: &JsValue,
    label: &str,
    source: &str,
    entries: &[wgpu::BindGroupLayoutEntry],
) -> Result<js_sys::Promise, JsValue> {
    let module = call(
        device,
        "createShaderModule",
        &object(&[("label", label.into()), ("code", source.into())])?,
    )?;
    let bindings = Array::new();
    for entry in entries {
        bindings.push(&binding(entry)?);
    }
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
    call(device, "createComputePipelineAsync", &descriptor)?.dyn_into()
}

/// A binding as the browser takes it: a Kernel binds buffers alone.
fn binding(entry: &wgpu::BindGroupLayoutEntry) -> Result<JsValue, JsValue> {
    let wgpu::BindingType::Buffer {
        ty,
        has_dynamic_offset,
        min_binding_size,
    } = entry.ty
    else {
        return Err("a Kernel binds buffers alone".into());
    };
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
    ])?;
    object(&[
        ("binding", entry.binding.into()),
        ("visibility", entry.visibility.bits().into()),
        ("buffer", buffer),
    ])
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
