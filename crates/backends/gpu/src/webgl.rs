//! Synchronous WebGL2 driver for a page or mini-game (`globalThis.entelechy_webgl`).
//!
//! Runs `Mode::Map` Kernels as `#version 300 es` fragment passes over `RGBA32UI`
//! word-stream textures without `WebAssembly.Suspending` (JSPI); non-map and
//! unsupported Kernels decline to the ordered backend.

use std::any::Any;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use entelechy_abi::{Domain, Handle, Invocation, MadeFrom, Scope, Version};
use entelechy_ir::{Mode, Validated};
use entelechy_rules::placement::{Placement, placement};
use entelechy_rules::{
    ArithmeticError, Backend, ExecutionError, NativeOp, Prepared, check_invocation,
};
use js_sys::{Function, Object, Reflect, Uint8Array, Uint32Array};
use wasm_bindgen::{JsCast, JsValue};

use crate::glsl::{self, GlslPass};
use crate::pool::{Held, SPARE_BUFFERS, SPARE_BYTES};
use crate::present::Overlay;
use crate::{GpuError, wgsl};

/// Whether the host explicitly asked to run on WebGL2 (`entelechy_webgl_only`).
pub(crate) fn prefer_webgl() -> bool {
    Reflect::get(&js_sys::global(), &"entelechy_webgl_only".into()).is_ok_and(|v| v.is_truthy())
}

/// Whether this runtime has both a browser WebGPU adapter entry and JSPI
/// (`WebAssembly.Suspending`) so `WgpuBackend` can suspend without trapping.
pub(crate) fn can_park_webgpu() -> bool {
    let global = js_sys::global();
    let has_jspi = Reflect::get(&global, &"WebAssembly".into())
        .ok()
        .and_then(|wasm| Reflect::get(&wasm, &"Suspending".into()).ok())
        .is_some_and(|s| s.is_function());
    if !has_jspi {
        return false;
    }
    Reflect::get(&global, &"navigator".into())
        .ok()
        .filter(JsValue::is_object)
        .and_then(|nav| Reflect::get(&nav, &"gpu".into()).ok())
        .is_some_and(|gpu| gpu.is_object())
}

struct JsBridge {
    target: Object,
    open: Function,
    create_texture: Function,
    delete_texture: Function,
    upload_words: Function,
    copy_words: Function,
    read_words: Function,
    compile_program: Function,
    run_pass: Function,
    reduce_fail: Function,
    same_textures: Function,
    canvas_size: Function,
    present: Function,
    finish: Function,
}

impl JsBridge {
    fn resolve() -> Result<Self, String> {
        let target: Object = Reflect::get(&js_sys::global(), &"entelechy_webgl".into())
            .ok()
            .and_then(|v| v.dyn_into().ok())
            .ok_or_else(|| "no `entelechy_webgl` host on globalThis".to_owned())?;
        let method = |name: &str| -> Result<Function, String> {
            Reflect::get(&target, &name.into())
                .ok()
                .and_then(|v| v.dyn_into().ok())
                .ok_or_else(|| format!("`entelechy_webgl.{name}` is missing"))
        };
        Ok(Self {
            open: method("open")?,
            create_texture: method("createTexture")?,
            delete_texture: method("deleteTexture")?,
            upload_words: method("uploadWords")?,
            copy_words: method("copyWords")?,
            read_words: method("readWords")?,
            compile_program: method("compileProgram")?,
            run_pass: method("runPass")?,
            reduce_fail: method("reduceFail")?,
            same_textures: method("sameTextures")?,
            canvas_size: method("canvasSize")?,
            present: method("present")?,
            finish: method("finish")?,
            target,
        })
    }
}

thread_local! {
    static BRIDGE: std::cell::RefCell<Option<JsBridge>> = const { std::cell::RefCell::new(None) };
}

fn with_bridge<T>(f: impl FnOnce(&JsBridge) -> Result<T, String>) -> Result<T, String> {
    BRIDGE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(JsBridge::resolve()?);
        }
        f(slot.as_ref().expect("resolved"))
    })
}

fn js_err(err: JsValue) -> String {
    err.as_string().unwrap_or_else(|| format!("{err:?}"))
}

fn prop_u32(obj: &JsValue, name: &str, fallback: u32) -> u32 {
    Reflect::get(obj, &name.into())
        .ok()
        .and_then(|v| v.as_f64())
        .map_or(fallback, |n| n as u32)
}

/// A resident Field stored in a WebGL2 `RGBA32UI` texture.
pub(crate) struct WebGlImage {
    pub(crate) id: u32,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) bytes: u64,
    pub(crate) spare: Option<WebGlSpare>,
    pub(crate) holds: Option<Held>,
}

impl Drop for WebGlImage {
    fn drop(&mut self) {
        if let Some(spare) = self.spare.take() {
            spare.keep(
                self.id,
                self.width,
                self.height,
                self.bytes,
                self.holds.take(),
            );
        }
    }
}

#[derive(Clone)]
struct KeptTex {
    id: u32,
    width: u32,
    height: u32,
    bytes: u64,
    holds: Option<Held>,
    kept_at: u64,
}

/// Bounded pool of released `RGBA32UI` textures, matching [`crate::pool::Spare`].
#[derive(Clone, Default)]
pub(crate) struct WebGlSpare {
    held: Arc<Mutex<Vec<KeptTex>>>,
    evaluations: Arc<AtomicU64>,
    gone: Arc<Mutex<Vec<u32>>>,
}

impl WebGlSpare {
    fn take(&self, width: u32, height: u32) -> Result<(u32, u64), String> {
        let bytes = u64::from(width) * u64::from(height) * 16;
        {
            let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
            if let Some(index) = held
                .iter()
                .enumerate()
                .filter(|(_, tex)| tex.width == width && tex.height == height)
                .min_by_key(|(idx, tex)| (tex.holds.is_some(), std::cmp::Reverse(*idx)))
                .map(|(idx, _)| idx)
            {
                let taken = held.remove(index);
                return Ok((taken.id, taken.bytes));
            }
        }
        let id = with_bridge(|b| {
            let val = b
                .create_texture
                .call2(&b.target, &width.into(), &height.into())
                .map_err(js_err)?;
            let id = val.as_f64().unwrap_or(0.0) as u32;
            if id == 0 {
                Err("WebGL2 failed to allocate RGBA32UI texture".into())
            } else {
                Ok(id)
            }
        })?;
        Ok((id, bytes))
    }

    fn take_made(
        &self,
        made: &[MadeFrom],
        width: u32,
        height: u32,
    ) -> Option<(u32, u64, Vec<Range<usize>>)> {
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let spans_for = |holds: &Held| -> Option<Vec<Range<usize>>> {
            if let Some(mine) = made.iter().find(|mine| mine.words == holds.words) {
                return Some(mine.changed.clone());
            }
            made.iter().find_map(|mine| {
                let theirs = holds.made.iter().find(|t| t.words == mine.words)?;
                let mut spans: Vec<Range<usize>> = mine
                    .changed
                    .iter()
                    .chain(&theirs.changed)
                    .cloned()
                    .collect();
                spans.sort_by_key(|span| (span.start, span.end));
                spans.dedup();
                Some(spans)
            })
        };
        let (index, spans) = held
            .iter()
            .enumerate()
            .filter(|(_, tex)| tex.width == width && tex.height == height)
            .filter_map(|(idx, tex)| Some((idx, spans_for(tex.holds.as_ref()?)?)))
            .min_by_key(|(_, spans)| spans.iter().map(ExactSizeIterator::len).sum::<usize>())?;
        let taken = held.remove(index);
        Some((taken.id, taken.bytes, spans))
    }

    fn keep(&self, id: u32, width: u32, height: u32, bytes: u64, holds: Option<Held>) {
        if bytes > SPARE_BYTES {
            self.let_go(id);
            return;
        }
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let now = self.evaluations.load(Ordering::Relaxed);
        held.push(KeptTex {
            id,
            width,
            height,
            bytes,
            holds,
            kept_at: now,
        });
        let mut total: u64 = held.iter().map(|t| t.bytes).sum();
        while held.len() > SPARE_BUFFERS || total > SPARE_BYTES {
            let evicted = held.remove(0);
            total = total.saturating_sub(evicted.bytes);
            self.let_go(evicted.id);
        }
    }

    fn let_go(&self, id: u32) {
        self.gone.lock().unwrap_or_else(|p| p.into_inner()).push(id);
    }

    fn bury(&self) {
        let gone: Vec<u32> = self
            .gone
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .drain(..)
            .collect();
        if gone.is_empty() {
            return;
        }
        let _ = with_bridge(|b| {
            for id in gone {
                let _ = b.delete_texture.call1(&b.target, &id.into());
            }
            Ok(())
        });
    }

    fn evaluation_begun(&self) {
        let now = self.evaluations.fetch_add(1, Ordering::Relaxed) + 1;
        let mut held = self.held.lock().unwrap_or_else(|p| p.into_inner());
        let mut gone = self.gone.lock().unwrap_or_else(|p| p.into_inner());
        held.retain(|tex| {
            let fresh = tex.kept_at + 1 >= now;
            if !fresh {
                gone.push(tex.id);
            }
            fresh
        });
    }

    fn bytes(&self) -> u64 {
        self.held
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .map(|t| t.bytes)
            .sum()
    }
}

/// Prepared GLSL ES 3.00 passes for a `Mode::Map` Kernel, compiled lazily.
struct WebGlLazy {
    id: String,
    module: Validated,
    packed: Vec<usize>,
    passes: Vec<GlslPass>,
    unbuilt: Arc<Mutex<Vec<(Validated, String)>>>,
    built: OnceLock<Result<Vec<u32>, String>>,
}

impl Prepared for WebGlLazy {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl WebGlLazy {
    fn is_built(&self) -> bool {
        self.built.get().is_some()
    }

    fn built(&self) -> Result<&[u32], ExecutionError> {
        self.built
            .get_or_init(|| {
                let input_count = (self.module.manifest.inputs.len() - self.packed.len()) as u32;
                let has_packed = !self.packed.is_empty();
                let mut programs = Vec::with_capacity(self.passes.len());
                for pass in &self.passes {
                    let built = with_bridge(|b| {
                        let args = js_sys::Array::new();
                        args.push(&pass.source.as_str().into());
                        args.push(&(self.module.manifest.inputs.len() as u32).into());
                        let packed_js = Uint32Array::from(
                            self.packed
                                .iter()
                                .map(|&k| k as u32)
                                .collect::<Vec<u32>>()
                                .as_slice(),
                        );
                        args.push(&packed_js);
                        args.push(&has_packed.into());
                        args.push(&(pass.outputs.len() as u32).into());
                        args.push(&pass.fails.into());
                        let _ = input_count;
                        let val = b.compile_program.apply(&b.target, &args).map_err(js_err)?;
                        if let Some(id) = val.as_f64().filter(|&n| n > 0.0) {
                            Ok(id as u32)
                        } else {
                            Err(val
                                .as_string()
                                .unwrap_or_else(|| "shader compilation failed".into()))
                        }
                    });
                    match built {
                        Ok(prog) => programs.push(prog),
                        Err(error) => {
                            for prev in programs {
                                let _ = prev;
                            }
                            let last = error
                                .lines()
                                .rev()
                                .find(|l| !l.trim().is_empty())
                                .unwrap_or("")
                                .trim();
                            let short: String = last.chars().take(160).collect();
                            let mut unbuilt =
                                self.unbuilt.lock().unwrap_or_else(|p| p.into_inner());
                            if !unbuilt.iter().any(|(seen, _)| seen == &self.module) {
                                unbuilt.push((self.module.clone(), short));
                            }
                            return Err(format!(
                                "the device's own compiler could not build `{}`: {error}",
                                self.id
                            ));
                        }
                    }
                }
                Ok(programs)
            })
            .as_deref()
            .map_err(|e| ExecutionError::Device(e.clone()))
    }
}

#[derive(Default)]
struct WebGlScratch {
    context: Vec<u8>,
    params: Vec<u8>,
    words: Vec<u32>,
}

/// WebGL2 GPU backend running in a browser page or mini-game.
pub(crate) struct WebGlBackend {
    renderer: String,
    max_texture_size: u32,
    max_texture_units: u32,
    max_draw_buffers: u32,
    max_uniform_bytes: u32,
    lock: Mutex<()>,
    spare: WebGlSpare,
    staging: Mutex<WebGlScratch>,
    uploaded: AtomicU64,
    measured: AtomicBool,
    last_nanos: AtomicU64,
    unbuilt: Arc<Mutex<Vec<(Validated, String)>>>,
}

impl WebGlBackend {
    pub(crate) fn open() -> Result<Self, GpuError> {
        let info = with_bridge(|b| {
            let res = b.open.call0(&b.target).map_err(js_err)?;
            if res.is_null() || res.is_undefined() {
                return Err("WebGL2 is unavailable".into());
            }
            if let Some(err) = Reflect::get(&res, &"error".into())
                .ok()
                .and_then(|v| v.as_string())
                .filter(|s| !s.is_empty())
            {
                return Err(err);
            }
            Ok((
                Reflect::get(&res, &"renderer".into())
                    .ok()
                    .and_then(|v| v.as_string())
                    .unwrap_or_else(|| "WebGL2".into()),
                prop_u32(&res, "maxTextureSize", 2048),
                prop_u32(&res, "maxTextureUnits", 16),
                prop_u32(&res, "maxDrawBuffers", 4),
                prop_u32(&res, "maxUniformBlockBytes", 16384),
            ))
        })
        .map_err(GpuError)?;

        let (renderer, max_texture_size, max_texture_units, max_draw_buffers, max_uniform_bytes) =
            info;
        Ok(Self {
            renderer,
            max_texture_size: max_texture_size.max(glsl::ROW_TEXELS),
            max_texture_units: max_texture_units.max(8),
            max_draw_buffers: max_draw_buffers.max(1),
            max_uniform_bytes: max_uniform_bytes.max(16384),
            lock: Mutex::new(()),
            spare: WebGlSpare::default(),
            staging: Mutex::new(WebGlScratch::default()),
            uploaded: AtomicU64::new(0),
            measured: AtomicBool::new(false),
            last_nanos: AtomicU64::new(u64::MAX),
            unbuilt: Arc::new(Mutex::new(Vec::new())),
        })
    }

    pub(crate) fn adapter_name(&self) -> &str {
        &self.renderer
    }

    pub(crate) fn measured(&self) -> bool {
        self.measured.load(Ordering::Relaxed)
    }

    pub(crate) fn display(&self) -> WebGlDisplay {
        WebGlDisplay
    }

    fn packing(&self, module: &Validated) -> Vec<usize> {
        let inputs = module.manifest.inputs.len();
        let most = self.max_texture_units as usize;
        if inputs <= most {
            return Vec::new();
        }
        let over = inputs - most;
        let mut candidates: Vec<usize> = (0..inputs)
            .rev()
            .filter(|&k| module.manifest.inputs[k].shape() == entelechy_ir::Shape::Rows)
            .chain(
                (0..inputs)
                    .rev()
                    .filter(|&k| module.manifest.inputs[k].shape() != entelechy_ir::Shape::Rows),
            )
            .take(over + 1)
            .collect();
        candidates.sort_unstable();
        candidates
    }

    fn refusal(&self, module: &Validated) -> Option<String> {
        if module.manifest.mode != Mode::Map {
            return Some(format!(
                "a WebGL2 device runs map kernels only, not {}",
                module.manifest.mode.name()
            ));
        }
        if !module.manifest.memories.is_empty() {
            return Some("a WebGL2 device has no read-write storage memories".to_owned());
        }
        let braces = wgsl::brace_depth(module);
        if braces > wgsl::BRACE_LIMIT {
            return Some(format!(
                "its branches and loops nest {braces} deep, and a shader nests at most {}",
                wgsl::BRACE_LIMIT
            ));
        }
        if module.holds(entelechy_abi::Type::Double) {
            return Some("it holds a Double, which the device does not compute".to_owned());
        }
        if wgsl::filtered_uses(module).any {
            return Some(
                "it uses filtered or mipmap sampling, which WebGL2 delegates to the ordered backend"
                    .to_owned(),
            );
        }
        if module.holds(entelechy_abi::Type::Long) && !wgsl::pairs_take(module) {
            return Some(
                "it holds a Long vector or unsupported Long intrinsic on a 32-bit device"
                    .to_owned(),
            );
        }
        for port in module
            .manifest
            .inputs
            .iter()
            .chain(&module.manifest.outputs)
        {
            if port.ty.schema.uniform_encoding().is_none() && !port.ty.schema.is_wide() {
                return Some(format!(
                    "`{}` keeps its scalars in more than one narrow encoding",
                    port.name
                ));
            }
        }
        let packed = self.packing(module);
        let context_bytes = (wgsl::CONTEXT_HEAD + module.manifest.inputs.len()) * 16;
        let params_bytes = (wgsl::param_slots(module).1 + packed.len().div_ceil(4)).max(1) * 16;
        let max_ubo = self.max_uniform_bytes as usize;
        if context_bytes > max_ubo || params_bytes > max_ubo {
            return Some(format!(
                "its uniform blocks exceed the WebGL2 limit of {max_ubo} bytes"
            ));
        }
        None
    }

    fn write_words(
        &self,
        tex_id: u32,
        start_word: usize,
        words: &[u32],
    ) -> Result<(), ExecutionError> {
        if words.is_empty() {
            return Ok(());
        }
        const CHUNK_WORDS: usize = 1 << 18;
        for (k, piece) in words.chunks(CHUNK_WORDS).enumerate() {
            let at = (start_word + k * CHUNK_WORDS) as u32;
            let view = Uint32Array::from(piece);
            with_bridge(|b| {
                b.upload_words
                    .call3(&b.target, &tex_id.into(), &at.into(), &view)
                    .map(drop)
                    .map_err(js_err)
            })
            .map_err(ExecutionError::Device)?;
            self.uploaded
                .fetch_add((piece.len() * 4) as u64, Ordering::Relaxed);
        }
        Ok(())
    }

    fn bind_input(&self, scope: &mut Scope<'_>, handle: Handle) -> Result<u32, ExecutionError> {
        if let Some(id) = scope
            .resident(handle)?
            .and_then(|dev| dev.downcast_ref::<WebGlImage>())
            .map(|img| img.id)
        {
            return Ok(id);
        }
        let len = scope.storage(handle)?.len();
        let (width, height) = glsl::texel_shape(len);
        if width > self.max_texture_size || height > self.max_texture_size {
            return Err(ExecutionError::Device(format!(
                "Field texture {width}x{height} exceeds max texture size {}",
                self.max_texture_size
            )));
        }
        let made = scope.made_from(handle)?;
        let found = self.spare.take_made(&made, width, height).or_else(|| {
            made.iter().find_map(|from| {
                self.copied(scope, from.words, width, height)
                    .map(|(id, bytes)| (id, bytes, from.changed.clone()))
            })
        });
        let (id, bytes, spans) = match found {
            Some(found) => found,
            None => {
                let (id, bytes) = self
                    .spare
                    .take(width, height)
                    .map_err(ExecutionError::Device)?;
                (id, bytes, std::iter::once(0..len).collect())
            }
        };
        let storage = scope.storage(handle)?;
        for span in spans {
            if let Err(err) = self.write_words(id, span.start, &storage[span]) {
                self.spare.let_go(id);
                return Err(err);
            }
        }
        let holds = Held {
            words: scope.version(handle)?,
            made,
        };
        scope.make_resident(
            handle,
            Box::new(WebGlImage {
                id,
                width,
                height,
                bytes,
                spare: Some(self.spare.clone()),
                holds: Some(holds),
            }),
        )?;
        Ok(id)
    }

    fn copied(
        &self,
        scope: &Scope<'_>,
        words: Version,
        width: u32,
        height: u32,
    ) -> Option<(u32, u64)> {
        if !scope.holds(words).ok()? {
            return None;
        }
        let from = scope
            .resident(words.field())
            .ok()??
            .downcast_ref::<WebGlImage>()?;
        if from.width != width || from.height != height {
            return None;
        }
        let src_id = from.id;
        let (dst_id, bytes) = self.spare.take(width, height).ok()?;
        let ok = with_bridge(|b| {
            let args = js_sys::Array::new();
            args.push(&src_id.into());
            args.push(&0u32.into());
            args.push(&dst_id.into());
            args.push(&0u32.into());
            args.push(&(width * height * 4).into());
            b.copy_words
                .apply(&b.target, &args)
                .map(drop)
                .map_err(js_err)
        });
        if ok.is_err() {
            self.spare.let_go(dst_id);
            return None;
        }
        Some((dst_id, bytes))
    }

    fn fill_uniforms(
        &self,
        module: &Validated,
        invocation: &Invocation,
        domain: Domain,
        input_extents: &[Domain],
        offsets: &[u32],
        run_work_limit: u32,
    ) -> (Uint8Array, Uint8Array) {
        let mut scratch = self.staging.lock().unwrap_or_else(|p| p.into_inner());
        let WebGlScratch {
            context,
            params,
            words,
        } = &mut *scratch;
        context.clear();
        context.reserve((wgsl::CONTEXT_HEAD + input_extents.len()) * 16);
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
        push_f32(context, invocation.context.scale);
        push_u32(context, 0);
        push_f32(context, 1.0);
        push_u32(context, run_work_limit);
        let sticks = input.sticks;
        push_f32(context, sticks[0][0]);
        push_f32(context, sticks[0][1]);
        push_f32(context, sticks[1][0]);
        push_f32(context, sticks[1][1]);
        push_f32(context, sticks[2][0]);
        push_f32(context, sticks[2][1]);
        push_f32(context, input.pads as f32);
        push_u32(context, invocation.context.queued.min(i32::MAX as u32));
        for &word in &input.typed_words() {
            push_u32(context, word);
        }
        for extent in input_extents {
            push_u32(context, extent.width());
            push_u32(context, extent.height());
            push_u32(context, extent.depth());
            push_u32(context, 0);
        }

        params.clear();
        for value in &invocation.params {
            words.clear();
            value.push_words(words);
            let padded = words.len().div_ceil(4).max(1) * 4;
            words.resize(padded, 0);
            for &word in words.iter() {
                params.extend_from_slice(&word.to_le_bytes());
            }
        }
        if params.is_empty() {
            params.resize(16, 0);
        }
        if !offsets.is_empty() {
            params.truncate(wgsl::param_slots(module).1 * 16);
            for offset in offsets {
                params.extend_from_slice(&offset.to_le_bytes());
            }
            params.resize(params.len().div_ceil(16).max(1) * 16, 0);
        }
        (
            Uint8Array::from(context.as_slice()),
            Uint8Array::from(params.as_slice()),
        )
    }

    fn input_values(&self, scope: &Scope<'_>, handle: Handle) -> Result<Vec<u32>, ExecutionError> {
        if let Ok(values) = scope.values(handle) {
            return Ok(values.into_owned());
        }
        let dev = scope
            .resident(handle)?
            .ok_or_else(|| ExecutionError::Device("input holds no words".into()))?;
        let schema = scope.schema(handle)?;
        let count = scope.size(handle)?.count();
        let storage = self
            .download(dev, schema.storage_words(count) * 4)
            .map_err(ExecutionError::Device)?;
        Ok(schema.unpack(&storage, count))
    }

    fn adopt_words(
        &self,
        scope: &mut Scope<'_>,
        schema: entelechy_abi::Schema,
        domain: Domain,
        values: &[u32],
    ) -> Result<Handle, ExecutionError> {
        let output = scope.create_for_fill(schema.clone(), domain)?;
        if let Err(err) = scope.write_values(output, values) {
            let _ = scope.release(output);
            return Err(err.into());
        }
        // Upload so downstream GPU Kernels read it directly as resident.
        let _ = self.bind_input(scope, output);
        Ok(output)
    }
}

impl Backend for WebGlBackend {
    fn name(&self) -> &'static str {
        "gpu"
    }

    fn repeats_exactly(&self) -> bool {
        true
    }

    fn has_device(&self) -> bool {
        true
    }

    fn has_compute_device(&self) -> bool {
        false
    }

    fn set_measured(&self, on: bool) {
        self.measured.store(on, Ordering::Relaxed);
    }

    fn last_device_time(&self) -> Option<std::time::Duration> {
        let nanos = self.last_nanos.swap(u64::MAX, Ordering::Relaxed);
        (nanos != u64::MAX).then(|| std::time::Duration::from_nanos(nanos))
    }

    fn settle_device(&self) {
        let _ = with_bridge(|b| b.finish.call0(&b.target).map(drop).map_err(js_err));
    }

    fn evaluation_begun(&self) {
        self.spare.evaluation_begun();
        self.spare.bury();
    }

    fn memory(&self) -> Vec<(&'static str, u64)> {
        vec![("spare textures", self.spare.bytes())]
    }

    fn uploaded(&self) -> u64 {
        self.uploaded.load(Ordering::Relaxed)
    }

    fn upload(&self, handle: Handle, scope: &mut Scope<'_>) -> Result<bool, ExecutionError> {
        self.bind_input(scope, handle).map(|_| true)
    }

    fn join(
        &self,
        parts: &[(Handle, Range<usize>)],
        schema: &entelechy_abi::Schema,
        domain: Domain,
        scope: &mut Scope<'_>,
    ) -> Result<Option<Handle>, ExecutionError> {
        let element_bytes = schema.element_bytes();
        if !element_bytes.is_multiple_of(4) {
            return Ok(None);
        }
        let words_per = element_bytes / 4;
        let mut pieces = Vec::with_capacity(parts.len());
        for (part, range) in parts {
            let from = range.start * words_per;
            let count = range.len() * words_per;
            pieces.push((self.bind_input(scope, *part)?, from, count));
        }
        let total_words: usize = pieces.iter().map(|(_, _, count)| *count).sum();
        if total_words != schema.storage_words(domain.count()) {
            return Err(ExecutionError::Device(format!(
                "the parts joined hold {} bytes, where {} over {domain} takes {}",
                total_words * 4,
                schema,
                schema.storage_words(domain.count()) * 4
            )));
        }
        let (tw, th) = glsl::texel_shape(total_words);
        if tw > self.max_texture_size || th > self.max_texture_size {
            return Ok(None);
        }
        scope.can_hold(&[(schema, domain)])?;
        let (dst_id, bytes) = self.spare.take(tw, th).map_err(ExecutionError::Device)?;
        let mut at = 0usize;
        for (src_id, from, count) in pieces {
            if count > 0 {
                let res = with_bridge(|b| {
                    let args = js_sys::Array::new();
                    args.push(&(src_id).into());
                    args.push(&(from as u32).into());
                    args.push(&(dst_id).into());
                    args.push(&(at as u32).into());
                    args.push(&(count as u32).into());
                    b.copy_words
                        .apply(&b.target, &args)
                        .map(drop)
                        .map_err(js_err)
                });
                if let Err(err) = res {
                    self.spare.let_go(dst_id);
                    self.spare.bury();
                    return Err(ExecutionError::Device(err));
                }
            }
            at += count;
        }
        let dev = Box::new(WebGlImage {
            id: dst_id,
            width: tw,
            height: th,
            bytes,
            spare: Some(self.spare.clone()),
            holds: None,
        });
        Ok(Some(scope.adopt_resident(schema.clone(), domain, dev)?))
    }

    fn declines(&self, module: &Validated) -> Option<String> {
        self.refusal(module).or_else(|| {
            let unbuilt = self.unbuilt.lock().unwrap_or_else(|p| p.into_inner());
            unbuilt
                .iter()
                .find(|(seen, _)| seen == module)
                .map(|(_, why)| format!("the device's own compiler could not build it ({why})"))
        })
    }

    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        let carried = entelechy_ir::carried_builders(module)
            .map(entelechy_ir::validate)
            .transpose()
            .map_err(|e| format!("`{}`: {e}", module.manifest.id))?;
        let module = carried.as_ref().unwrap_or(module);
        if let Some(reason) = self.refusal(module) {
            return Err(format!(
                "`{}`: {reason}; a session with an ordered backend runs it there",
                module.manifest.id
            ));
        }
        let packed = self.packing(module);
        let passes = glsl::emit_passes(module, &packed, self.max_draw_buffers as usize);
        Ok(Box::new(WebGlLazy {
            id: module.manifest.id.to_string(),
            module: module.clone(),
            packed,
            passes,
            unbuilt: Arc::clone(&self.unbuilt),
            built: OnceLock::new(),
        }))
    }

    fn finish(&self, prepared: &dyn Prepared) {
        if let Some(lazy) = prepared.as_any().downcast_ref::<WebGlLazy>() {
            let _ = lazy.built();
        }
    }

    fn finished(&self, prepared: &dyn Prepared) -> bool {
        prepared
            .as_any()
            .downcast_ref::<WebGlLazy>()
            .is_none_or(WebGlLazy::is_built)
    }

    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        check_invocation(module, invocation, scope)?;
        let lazy = prepared
            .as_any()
            .downcast_ref::<WebGlLazy>()
            .ok_or_else(|| {
                ExecutionError::Device("kernel was prepared by a different backend".into())
            })?;
        let programs = lazy.built()?;
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let domain = invocation.domain;
        let element_count = domain.count();

        let fields: Vec<_> = module
            .manifest
            .outputs
            .iter()
            .map(|port| (&port.ty.schema, domain))
            .collect();
        scope.can_hold(&fields)?;

        let mut input_ids = Vec::with_capacity(invocation.inputs.len());
        let mut input_extents = Vec::with_capacity(invocation.inputs.len());
        for &handle in &invocation.inputs {
            input_extents.push(scope.size(handle)?);
            input_ids.push(self.bind_input(scope, handle)?);
        }

        let mut offsets = Vec::new();
        let mut packed_id = 0u32;
        if !lazy.packed.is_empty() {
            let mut total_words = 0usize;
            let mut pieces = Vec::with_capacity(lazy.packed.len());
            for &idx in &lazy.packed {
                let words = module.manifest.inputs[idx]
                    .ty
                    .schema
                    .storage_words(input_extents[idx].count());
                let offset = u32::try_from(total_words).map_err(|_| {
                    ExecutionError::Device("packed inputs exceed 2^32 words".into())
                })?;
                offsets.push(offset);
                pieces.push((input_ids[idx], offset, words as u32));
                total_words += words;
            }
            let (pw, ph) = glsl::texel_shape(total_words);
            let (pid, _) = self.spare.take(pw, ph).map_err(ExecutionError::Device)?;
            packed_id = pid;
            for (src_id, dst_word, count) in pieces {
                if count == 0 {
                    continue;
                }
                if let Err(err) = with_bridge(|b| {
                    let args = js_sys::Array::new();
                    args.push(&src_id.into());
                    args.push(&0u32.into());
                    args.push(&packed_id.into());
                    args.push(&dst_word.into());
                    args.push(&count.into());
                    b.copy_words
                        .apply(&b.target, &args)
                        .map(drop)
                        .map_err(js_err)
                }) {
                    self.spare.let_go(packed_id);
                    return Err(ExecutionError::Device(err));
                }
            }
        }

        let run_limit = scope.work().min(u64::from(u32::MAX - 1)) as u32;
        let (ctx_js, params_js) = self.fill_uniforms(
            module,
            invocation,
            domain,
            &input_extents,
            &offsets,
            run_limit,
        );
        let in_js = Uint32Array::from(input_ids.as_slice());

        let mut out_textures: Vec<(u32, u32, u32, u64)> =
            Vec::with_capacity(module.manifest.outputs.len());
        for port in &module.manifest.outputs {
            let words = port.ty.schema.storage_words(element_count);
            let (tw, th) = glsl::texel_shape(words);
            if tw > self.max_texture_size || th > self.max_texture_size {
                for (id, ..) in out_textures {
                    self.spare.let_go(id);
                }
                if packed_id != 0 {
                    self.spare.let_go(packed_id);
                }
                return Err(ExecutionError::Device(format!(
                    "output texture {tw}x{th} exceeds max texture size {}",
                    self.max_texture_size
                )));
            }
            match self.spare.take(tw, th) {
                Ok((id, bytes)) => out_textures.push((id, tw, th, bytes)),
                Err(err) => {
                    for (id, ..) in out_textures {
                        self.spare.let_go(id);
                    }
                    if packed_id != 0 {
                        self.spare.let_go(packed_id);
                    }
                    return Err(ExecutionError::Device(err));
                }
            }
        }

        let measured = self.measured.load(Ordering::Relaxed);
        let started = measured.then(entelechy_rules::clock::now);
        let mut fail_result: Option<[u32; 4]> = None;

        for (pass, &prog) in lazy.passes.iter().zip(programs.iter()) {
            let (_, tw, th, _) = out_textures[pass.outputs[0]];
            let pass_outs: Vec<u32> = pass.outputs.iter().map(|&k| out_textures[k].0).collect();
            let outs_js = Uint32Array::from(pass_outs.as_slice());
            let fail_tex = if pass.fails {
                Some(self.spare.take(tw, th).map_err(ExecutionError::Device)?)
            } else {
                None
            };
            let fail_id = fail_tex.map_or(0, |(id, _)| id);

            let ran = with_bridge(|b| {
                let args = js_sys::Array::new();
                args.push(&prog.into());
                args.push(&tw.into());
                args.push(&th.into());
                args.push(&ctx_js);
                args.push(&params_js);
                args.push(&in_js);
                args.push(&packed_id.into());
                args.push(&outs_js);
                args.push(&fail_id.into());
                let res = b.run_pass.apply(&b.target, &args).map_err(js_err)?;
                if let Some(err) = res.as_string().filter(|s| !s.is_empty()) {
                    return Err(err);
                }
                if pass.fails {
                    let reduced: Uint32Array = b
                        .reduce_fail
                        .call3(&b.target, &fail_id.into(), &tw.into(), &th.into())
                        .map_err(js_err)?
                        .dyn_into()
                        .map_err(js_err)?;
                    let mut four = [0u32; 4];
                    reduced.copy_to(&mut four);
                    return Ok(Some(four));
                }
                Ok(None)
            });

            if let Some((fid, fbytes)) = fail_tex {
                self.spare.keep(fid, tw, th, fbytes, None);
            }
            match ran {
                Ok(Some(four)) => fail_result = Some(four),
                Ok(None) => {}
                Err(err) => {
                    for (id, ..) in out_textures {
                        self.spare.let_go(id);
                    }
                    if packed_id != 0 {
                        self.spare.let_go(packed_id);
                    }
                    self.spare.bury();
                    return Err(ExecutionError::Device(err));
                }
            }
        }

        if packed_id != 0 {
            let (pw, ph) = glsl::texel_shape(1);
            let _ = (pw, ph);
            self.spare.let_go(packed_id);
        }
        if let Some(started) = started {
            self.settle_device();
            let nanos = entelechy_rules::clock::since(started)
                .as_nanos()
                .min(u128::from(u64::MAX - 1)) as u64;
            self.last_nanos.store(nanos, Ordering::Relaxed);
        }
        self.spare.bury();

        if let Some([word, work_lo, work_hi, _]) = fail_result {
            let check_err = if word != 0 {
                let decoded = u32::MAX - word;
                let element = decoded / 4;
                let width = domain.width().max(1);
                let (x, y) = (element % width, element / width);
                Some(match decoded % 4 {
                    wgsl::FAILED_BUDGET => ExecutionError::Exhausted {
                        x,
                        y,
                        budget: module.manifest.budget,
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
            } else {
                let total_work = u64::from(work_lo) | (u64::from(work_hi) << 32);
                (total_work > scope.work()).then(|| ExecutionError::Overrun { work: scope.work() })
            };
            if let Some(err) = check_err {
                for (id, tw, th, bytes) in out_textures {
                    self.spare.keep(id, tw, th, bytes, None);
                }
                return Err(err);
            }
        }

        let mut handles = Vec::with_capacity(out_textures.len());
        for (port, (id, width, height, bytes)) in module.manifest.outputs.iter().zip(out_textures) {
            let dev = Box::new(WebGlImage {
                id,
                width,
                height,
                bytes,
                spare: Some(self.spare.clone()),
                holds: None,
            });
            match scope.adopt_resident(port.ty.schema.clone(), domain, dev) {
                Ok(h) => handles.push(h),
                Err(err) => {
                    for prev in handles {
                        let _ = scope.release(prev);
                    }
                    return Err(err.into());
                }
            }
        }
        Ok(handles)
    }

    fn execute_repeated(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
        repeat: u32,
        chain: (usize, usize),
    ) -> Option<Result<Vec<Handle>, ExecutionError>> {
        let (chain_in, chain_out) = chain;
        let lazy = prepared.as_any().downcast_ref::<WebGlLazy>()?;
        if repeat < 2
            || chain_out != 0
            || !lazy.packed.is_empty()
            || lazy.passes.len() != 1
            || module.manifest.mode != Mode::Map
            || module.manifest.outputs.len() != 1
            || self.measured.load(Ordering::Relaxed)
            || wgsl::can_fail(module)
        {
            return None;
        }
        let domain = invocation.domain;
        if scope.size(*invocation.inputs.get(chain_in)?).ok()? != domain {
            return None;
        }
        Some((|| {
            check_invocation(module, invocation, scope)?;
            let programs = lazy.built()?;
            let prog = programs[0];
            let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
            let schema = &module.manifest.outputs[0].ty.schema;
            scope.can_hold(&[(schema, domain), (schema, domain)])?;

            let mut input_ids = Vec::with_capacity(invocation.inputs.len());
            let mut input_extents = Vec::with_capacity(invocation.inputs.len());
            for &handle in &invocation.inputs {
                input_extents.push(scope.size(handle)?);
                input_ids.push(self.bind_input(scope, handle)?);
            }
            let (tw, th) = glsl::texel_shape(schema.storage_words(domain.count()));
            let (tex_a, bytes_a) = self.spare.take(tw, th).map_err(ExecutionError::Device)?;
            let (tex_b, bytes_b) = match self.spare.take(tw, th) {
                Ok(pair) => pair,
                Err(err) => {
                    self.spare.let_go(tex_a);
                    return Err(ExecutionError::Device(err));
                }
            };
            let run_limit = scope.work().min(u64::from(u32::MAX - 1)) as u32;
            let (ctx_js, params_js) =
                self.fill_uniforms(module, invocation, domain, &input_extents, &[], run_limit);
            let mut cur_out = (tex_a, bytes_a);
            let mut next_out = (tex_b, bytes_b);
            for step in 0..repeat {
                if step > 0 {
                    input_ids[chain_in] = next_out.0;
                }
                let in_js = Uint32Array::from(input_ids.as_slice());
                let outs_js = Uint32Array::from(&[cur_out.0][..]);
                let ran = with_bridge(|b| {
                    let args = js_sys::Array::new();
                    args.push(&prog.into());
                    args.push(&tw.into());
                    args.push(&th.into());
                    args.push(&ctx_js);
                    args.push(&params_js);
                    args.push(&in_js);
                    args.push(&0u32.into());
                    args.push(&outs_js);
                    args.push(&0u32.into());
                    let res = b.run_pass.apply(&b.target, &args).map_err(js_err)?;
                    if let Some(err) = res.as_string().filter(|s| !s.is_empty()) {
                        Err(err)
                    } else {
                        Ok(())
                    }
                });
                if let Err(err) = ran {
                    self.spare.let_go(cur_out.0);
                    self.spare.let_go(next_out.0);
                    self.spare.bury();
                    return Err(ExecutionError::Device(err));
                }
                std::mem::swap(&mut cur_out, &mut next_out);
            }
            // After `repeat` swaps, `next_out` holds the last written output and `cur_out` is spare.
            self.spare.keep(cur_out.0, tw, th, cur_out.1, None);
            self.spare.bury();
            let dev = Box::new(WebGlImage {
                id: next_out.0,
                width: tw,
                height: th,
                bytes: next_out.1,
                spare: Some(self.spare.clone()),
                holds: None,
            });
            Ok(vec![scope.adopt_resident(schema.clone(), domain, dev)?])
        })())
    }

    fn execute_native(
        &self,
        op: NativeOp,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, ExecutionError> {
        use entelechy_rules::raster;
        let domain = invocation.domain;
        match op {
            NativeOp::Rasterize => {
                let (v, t) = raster::check_rasterize(invocation, scope)?;
                let words = raster::rasterize_words(
                    &self.input_values(scope, v)?,
                    &self.input_values(scope, t)?,
                    domain.width(),
                    domain.height(),
                );
                Ok(vec![self.adopt_words(
                    scope,
                    raster::surface_schema(),
                    domain,
                    &words,
                )?])
            }
            NativeOp::RasterizeCanvas => {
                let (v, t) = raster::check_rasterize_canvas(invocation, scope)?;
                let words = raster::rasterize_words(
                    &self.input_values(scope, v)?,
                    &self.input_values(scope, t)?,
                    domain.width(),
                    domain.height(),
                );
                Ok(vec![self.adopt_words(
                    scope,
                    raster::surface_schema(),
                    domain,
                    &words,
                )?])
            }
            NativeOp::RasterizeCutout => {
                let (v, t, mask, alpha) = raster::check_rasterize_cutout(invocation, scope)?;
                let mask_size = scope.size(mask)?;
                let mask_words = self.input_values(scope, mask)?;
                let words = raster::rasterize_masked(
                    &self.input_values(scope, v)?,
                    &self.input_values(scope, t)?,
                    domain.width(),
                    domain.height(),
                    Some(raster::Cutout {
                        words: &mask_words,
                        width: mask_size.width(),
                        height: mask_size.height(),
                        alpha,
                    }),
                );
                Ok(vec![self.adopt_words(
                    scope,
                    raster::surface_schema(),
                    domain,
                    &words,
                )?])
            }
            NativeOp::Cover => {
                let (v, t, keys) = raster::check_cover(invocation, scope)?;
                let surface = raster::rasterize_words(
                    &self.input_values(scope, v)?,
                    &self.input_values(scope, t)?,
                    domain.width(),
                    domain.height(),
                );
                let words = raster::cover_words(&surface, &self.input_values(scope, keys)?);
                Ok(vec![self.adopt_words(
                    scope,
                    raster::canvas_schema(),
                    domain,
                    &words,
                )?])
            }
            NativeOp::Trace => {
                let inp = entelechy_rules::trace::check_trace(invocation, scope)?;
                let words = entelechy_rules::trace::trace_words(
                    &self.input_values(scope, inp.rays)?,
                    &self.input_values(scope, inp.vertices)?,
                    &self.input_values(scope, inp.triangles)?,
                    inp.flags,
                );
                Ok(vec![self.adopt_words(
                    scope,
                    entelechy_rules::trace::hit_schema(),
                    domain,
                    &words,
                )?])
            }
            NativeOp::TraceInstances => {
                let inp = entelechy_rules::trace::check_trace_instances(invocation, scope)?;
                let words = entelechy_rules::trace::trace_instances_words(
                    &self.input_values(scope, inp.rays)?,
                    &self.input_values(scope, inp.vertices)?,
                    &self.input_values(scope, inp.triangles)?,
                    &self.input_values(scope, inp.meshes)?,
                    &self.input_values(scope, inp.instances)?,
                    inp.flags,
                    inp.mask,
                );
                Ok(vec![self.adopt_words(
                    scope,
                    entelechy_rules::trace::hit_schema(),
                    domain,
                    &words,
                )?])
            }
        }
    }

    fn download(
        &self,
        device: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<Vec<u32>, String> {
        let image = device
            .downcast_ref::<WebGlImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let word_len = byte_len.div_ceil(4) as u32;
        if word_len == 0 {
            return Ok(Vec::new());
        }
        with_bridge(|b| {
            let arr: Uint32Array = b
                .read_words
                .call2(&b.target, &image.id.into(), &word_len.into())
                .map_err(js_err)?
                .dyn_into()
                .map_err(js_err)?;
            Ok(arr.to_vec())
        })
    }

    fn same_resident(
        &self,
        a: &(dyn Any + Send + Sync),
        b: &(dyn Any + Send + Sync),
        byte_len: usize,
    ) -> Result<bool, String> {
        let (Some(a), Some(b)) = (
            a.downcast_ref::<WebGlImage>(),
            b.downcast_ref::<WebGlImage>(),
        ) else {
            return Err("image is resident on a different backend".into());
        };
        if a.id == b.id {
            return Ok(true);
        }
        let _guard = self.lock.lock().unwrap_or_else(|p| p.into_inner());
        let words = u32::try_from(byte_len / 4).map_err(|_| "a Field too long to compare")?;
        if words == 0 {
            return Ok(true);
        }
        with_bridge(|br| {
            let same = br
                .same_textures
                .call3(&br.target, &a.id.into(), &b.id.into(), &words.into())
                .map_err(js_err)?;
            Ok(same.is_truthy())
        })
    }
}

/// WebGL2 zero-copy display handle.
#[derive(Clone, Copy)]
pub(crate) struct WebGlDisplay;

/// Target canvas bound by [`WebGlDisplay`].
pub(crate) struct WebGlPresenter {
    canvas: String,
}

impl WebGlDisplay {
    pub(crate) fn canvas_size(&self, canvas: &str) -> Result<(u32, u32), String> {
        with_bridge(|b| {
            let arr: Uint32Array = b
                .canvas_size
                .call1(&b.target, &canvas.into())
                .map_err(js_err)?
                .dyn_into()
                .map_err(js_err)?;
            if arr.length() < 2 {
                return Err(format!("no canvas `{canvas}` for WebGL2 presentation"));
            }
            let mut dims = [0u32; 2];
            arr.subarray(0, 2).copy_to(&mut dims);
            Ok((dims[0], dims[1]))
        })
    }

    pub(crate) fn presenter(&self, canvas: &str) -> WebGlPresenter {
        WebGlPresenter {
            canvas: canvas.to_owned(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn present(
        &self,
        presenter: &mut WebGlPresenter,
        device: &(dyn Any + Send + Sync),
        width: u32,
        height: u32,
        window_width: u32,
        window_height: u32,
        overlay: Option<Overlay<'_>>,
    ) -> Result<Option<Placement>, String> {
        let image = device
            .downcast_ref::<WebGlImage>()
            .ok_or_else(|| "image is resident on a different backend".to_string())?;
        if window_width == 0 || window_height == 0 {
            return Ok(None);
        }
        let placed = placement(width, height, window_width, window_height);
        let frame: [f32; 12] = [
            window_width as f32,
            window_height as f32,
            width as f32,
            height as f32,
            placed.x as f32,
            placed.y as f32,
            placed.width as f32,
            placed.height as f32,
            0.0,
            overlay.map_or(0.0, |o| o.width as f32),
            overlay.map_or(0.0, |o| o.height as f32),
            0.0,
        ];
        let drawn = with_bridge(|b| {
            let frame_js = js_sys::Float32Array::from(&frame[..]);
            let overlay_js: JsValue = match overlay {
                Some(ov) if !ov.pixels.is_empty() => Uint32Array::from(ov.pixels).into(),
                _ => JsValue::NULL,
            };
            let args = js_sys::Array::new();
            args.push(&presenter.canvas.as_str().into());
            args.push(&image.id.into());
            args.push(&window_width.into());
            args.push(&window_height.into());
            args.push(&frame_js);
            args.push(&overlay_js);
            let ok = b.present.apply(&b.target, &args).map_err(js_err)?;
            Ok(ok.is_truthy())
        })?;
        Ok(drawn.then_some(placed))
    }
}
