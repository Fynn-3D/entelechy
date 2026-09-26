//! Orla as a library in another process: the session the CLI and the
//! Player run, behind the C ABI in `include/orla.h`. A host opens a scene,
//! evaluates it under its own input, and reads Fields in place in the
//! Host's memory; the JSON protocol is reachable one line at a time.
//!
//! A session is used from one thread and one call at a time: a call made
//! while another is inside it is refused ([`OrlaSession`]). Returned
//! strings and pointers belong to the session until the next call on it;
//! a Field's words until the next evaluate, request or close. No panic
//! crosses the boundary: each entry reports one as an error.

use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::{CStr, CString, c_char, c_int};
use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use orla_abi::{Context, Handle, InputState, Shape};
use orla_session::Session;

/// The version of the header's contract, `ORLA_ABI_VERSION`.
pub const ABI_VERSION: u32 = 1;

/// What an evaluation sees; the layout of `OrlaInput` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrlaInput {
    pub time: f32,
    pub rate: f32,
    pub pointer: [f32; 2],
    pub wheel: [f32; 2],
    pub window: [f32; 2],
    pub scale: f32,
    pub buttons: u32,
    pub keys: [u32; 8],
}

impl Default for OrlaInput {
    fn default() -> Self {
        Self {
            time: 0.0,
            rate: 0.0,
            pointer: [0.0, 0.0],
            wheel: [0.0, 0.0],
            window: [0.0, 0.0],
            scale: 1.0,
            buttons: 0,
            keys: [0; 8],
        }
    }
}

/// A Field as it lies in memory; the layout of `OrlaField` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OrlaField {
    pub words: *const u32,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub count: u32,
    pub stride: u32,
    pub shape: u32,
    pub pixels: u32,
    pub schema: *const c_char,
}

/// A session as a caller holds it, and whether a call is inside it. A
/// call made while another is inside (a page reaching in while Orla waits
/// for its device) is refused as a NULL session is, so no two calls
/// touch what is held at once.
pub struct OrlaSession {
    inside: Cell<bool>,
    held: UnsafeCell<Held>,
}

/// A session entered by one call; what it holds is that call's alone
/// until it leaves.
struct Entered<'a>(&'a OrlaSession);

impl Deref for Entered<'_> {
    type Target = Held;

    fn deref(&self) -> &Held {
        // Only the call that entered reaches what is held ([`enter`]).
        unsafe { &*self.0.held.get() }
    }
}

impl DerefMut for Entered<'_> {
    fn deref_mut(&mut self) -> &mut Held {
        // Only the call that entered reaches what is held ([`enter`]).
        unsafe { &mut *self.0.held.get() }
    }
}

impl Drop for Entered<'_> {
    fn drop(&mut self) {
        self.0.inside.set(false);
    }
}

/// Enters a session for one call; `None` for a NULL session, or one
/// another call is inside.
///
/// # Safety
/// `session` is NULL or a live session.
unsafe fn enter<'a>(session: *const OrlaSession) -> Option<Entered<'a>> {
    let session = unsafe { session.as_ref() }?;
    if session.inside.replace(true) {
        return None;
    }
    Some(Entered(session))
}

/// A running scene and the buffers its answers live in.
struct Held {
    session: Session,
    error: CString,
    report: CString,
    response: CString,
    schema: CString,
    /// A narrow Field's words widened to the header's 32-bit words.
    widened: Vec<u32>,
    /// The GPU device as a canvas reaches it, on a GPU backend.
    #[cfg_attr(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        allow(dead_code)
    )]
    display: Option<orla_gpu::present::Display>,
    /// The canvas id last presented on, and its surface.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    presenter: Option<(String, orla_gpu::present::Presenter)>,
    /// The scene played: see `play.rs`.
    play: orla_session::play::Play,
    /// The last frame's picture and sound, as the play names them.
    shown: Option<(Handle, Option<Handle>)>,
    /// The play's messages, for `orla_play_said`.
    said: CString,
    /// A frame drawn on the CPU, `0x00RRGGBB`, for `orla_play_pixels`.
    pixels: Vec<u32>,
    /// The last picture read back, and its size; shown again when a frame
    /// fails or a picture cannot be read back.
    picture: (Vec<[f32; 4]>, (u32, u32)),
}

mod play;
pub use play::*;

/// Where a presented picture landed (`None`: the canvas took no frame)
/// and its size.
type Presented = (Option<orla_session::play::Placement>, (u32, u32));

thread_local! {
    static OPEN_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

const VERSION: &CStr =
    match CStr::from_bytes_with_nul(concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes()) {
        Ok(version) => version,
        Err(_) => panic!("the package version holds no NUL"),
    };
const EMPTY: &CStr = c"";

fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
}

/// A C string argument as `&str`, `None` for NULL.
///
/// # Safety
/// `pointer` is NULL or a NUL-terminated string that outlives the call.
unsafe fn text<'a>(pointer: *const c_char) -> Option<&'a str> {
    if pointer.is_null() {
        None
    } else {
        unsafe { CStr::from_ptr(pointer) }.to_str().ok()
    }
}

impl Held {
    /// A graph output's Field, `name` `None` for the first.
    fn output(&self, name: Option<&str>) -> Result<Handle, String> {
        self.session.output(name)
    }

    /// Presents an output on the canvas `canvas`; true when drawn, which
    /// needs a picture the GPU holds as float pixels.
    fn present(&mut self, canvas: &str, name: Option<&str>) -> Result<bool, String> {
        let handle = self.output(name)?;
        Ok(self.present_on(canvas, handle, None)?.is_some())
    }

    /// Draws `handle` from the GPU on the page's canvas `canvas`, with
    /// `overlay` over its corner. `None` when it is no picture the device
    /// holds; else where it landed (`None` when the canvas took no frame)
    /// and its size.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    fn present_on(
        &mut self,
        canvas: &str,
        handle: Handle,
        overlay: Option<orla_gpu::present::Overlay<'_>>,
    ) -> Result<Option<Presented>, String> {
        use wasm_bindgen::JsCast;
        let Some(display) = &self.display else {
            return Ok(None);
        };
        let Some((device, width, height)) = self.session.on_device(handle) else {
            return Ok(None);
        };
        let element: web_sys::HtmlCanvasElement = web_sys::window()
            .and_then(|window| window.document())
            .and_then(|document| document.get_element_by_id(canvas))
            .ok_or_else(|| format!("no element `{canvas}` on the page"))?
            .dyn_into()
            .map_err(|_| format!("`{canvas}` is not a canvas"))?;
        let (window_width, window_height) = (element.width(), element.height());
        if self.presenter.as_ref().is_none_or(|(id, _)| id != canvas) {
            let target = wgpu::SurfaceTarget::Canvas(element);
            let presenter = display.presenter(target, window_width, window_height)?;
            self.presenter = Some((canvas.to_owned(), presenter));
        }
        let (_, presenter) = self.presenter.as_mut().expect("just made");
        let placed = display.present(
            presenter,
            device,
            width,
            height,
            window_width,
            window_height,
            overlay,
        )?;
        Ok(Some((placed, (width, height))))
    }

    /// Outside a page there is no canvas: an error.
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    fn present_on(
        &mut self,
        _canvas: &str,
        _handle: Handle,
        _overlay: Option<orla_gpu::present::Overlay<'_>>,
    ) -> Result<Option<Presented>, String> {
        Err("a picture is presented on a page's canvas; a program reads it with orla_output".into())
    }

    fn fail(&mut self, message: impl Into<String>) -> c_int {
        self.error = c_string(&message.into());
        -1
    }

    fn clear(&mut self) {
        if !self.error.as_bytes().is_empty() {
            self.error = CString::default();
        }
    }

    /// The words of a materialized Field and the layout around them.
    fn field(&mut self, handle: Handle, field: &mut OrlaField) -> Result<(), String> {
        let data = self
            .session
            .field(handle)
            .map_err(|error| error.to_string())?;
        let domain = data.domain();
        let schema = data.schema();
        // Wide words are read in place; a narrower Field is widened into
        // the session's buffer, so the header's contract (32-bit words, a
        // layout of `f`, `i` and `b`) always holds.
        let words = if schema.is_wide() {
            data.storage().as_ptr()
        } else {
            self.widened = data.values().into_owned();
            self.widened.as_ptr()
        };
        let values = orla_abi::Schema::new(schema.scalars().to_vec());
        self.schema = c_string(&values.to_string());
        *field = OrlaField {
            words,
            width: domain.width(),
            height: domain.height(),
            depth: domain.depth(),
            count: domain.count() as u32,
            stride: schema.value_words() as u32,
            shape: match domain.shape() {
                Shape::Rows => 0,
                Shape::Grid => 1,
                Shape::Volume => 2,
            },
            pixels: schema.is_pixel() as u32,
            schema: self.schema.as_ptr(),
        };
        Ok(())
    }
}

/// Runs an entry point, turning a panic into an error result.
fn guarded(session: &mut Held, body: impl FnOnce(&mut Held) -> c_int) -> c_int {
    match catch_unwind(AssertUnwindSafe(|| body(session))) {
        Ok(code) => code,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .map(|s| (*s).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic inside Orla".to_owned());
            session.fail(format!("panic: {message}"))
        }
    }
}

/// The library's version.
#[unsafe(no_mangle)]
pub extern "C" fn orla_version() -> *const c_char {
    VERSION.as_ptr()
}

/// Opens a scene on a backend; NULL with `orla_open_error` set on failure.
///
/// # Safety
/// `scene`, `backend` and `library` are NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_open(
    scene: *const c_char,
    backend: *const c_char,
    library: *const c_char,
) -> *mut OrlaSession {
    unsafe { open(None, scene, backend, library) }
}

// In a page, Orla waits for the device by parking: `orla.wait` suspends
// Orla (JavaScript Promise Integration) until the device calls `orla.wake`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "orla")]
unsafe extern "C" {
    fn wait();
    fn wake();
}

// A page's clock: `performance.now()`, in milliseconds.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "orla")]
unsafe extern "C" {
    fn now() -> f64;
}

/// This process's clock, in seconds.
fn clock() -> f64 {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        unsafe { now() / 1000.0 }
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
    }
}

/// How this process waits for a device.
fn waiting() -> orla_session::Wait {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        fn park() {
            unsafe { wait() }
        }
        fn woken() {
            unsafe { wake() }
        }
        orla_session::Wait::parks(park, woken)
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        orla_session::Wait::blocking()
    }
}

/// Opens a scene over `store`, the disk when `None`.
///
/// # Safety
/// As `orla_open`.
unsafe fn open(
    store: Option<std::sync::Arc<dyn orla_session::Store>>,
    scene: *const c_char,
    backend: *const c_char,
    library: *const c_char,
) -> *mut OrlaSession {
    let backend = unsafe { text(backend) }.unwrap_or("best");
    let library = unsafe { text(library) }.map(Path::new);
    match unsafe { text(scene) } {
        Some(scene) => opened(store, Opening::Scene(Path::new(scene), library), backend),
        None => {
            OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string("orla_open needs a scene path"));
            std::ptr::null_mut()
        }
    }
}

/// What a session opens on: a scene and a library, or a replay that names
/// both and is played.
enum Opening<'a> {
    Scene(&'a Path, Option<&'a Path>),
    Replay(&'a Path),
}

/// Opens a session; NULL with `orla_open_error` set on failure.
fn opened(
    store: Option<std::sync::Arc<dyn orla_session::Store>>,
    opening: Opening<'_>,
    backend: &str,
) -> *mut OrlaSession {
    let set_error = |message: String| {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string(&message));
        std::ptr::null_mut()
    };
    let made = catch_unwind(AssertUnwindSafe(|| {
        let store = store.unwrap_or_else(|| std::sync::Arc::new(orla_session::Disk));
        let (scene, library, replay) = match opening {
            Opening::Scene(scene, library) => {
                (scene.to_path_buf(), library.map(Path::to_path_buf), None)
            }
            Opening::Replay(path) => {
                let text = store
                    .read_text(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let base = path.parent().unwrap_or(Path::new(""));
                let replay = orla_session::replay::parse(&text, base)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                (replay.scene.clone(), replay.kernels.clone(), Some(replay))
            }
        };
        // Every session here answers `orla_request`, so its reads are held
        // from the start to the directories of the scene as first loaded,
        // plus what `orla_allow` allows.
        let loaded = Session::open_loaded(
            store,
            &scene,
            library.as_deref(),
            backend,
            orla_session::Limits::default(),
            waiting(),
            true,
        )?;
        let (mut session, display, passed_over) =
            (loaded.session, loaded.display, loaded.passed_over);
        // The first load publishes inline; later replacements are prepared
        // off the caller's thread so frames go on (in place in a page).
        session.set_background(true);
        let mut play = orla_session::play::Play::new(clock);
        for passed in passed_over {
            play.say(format!("backend passed over: {passed}"));
        }
        if let Some(replay) = &replay {
            play.replay(replay, &mut session);
        }
        Ok((session, display, play))
    }));
    match made {
        Ok(Ok((session, display, play))) => Box::into_raw(Box::new(OrlaSession {
            inside: Cell::new(false),
            held: UnsafeCell::new(Held {
                session,
                error: CString::default(),
                report: CString::default(),
                response: CString::default(),
                schema: CString::default(),
                widened: Vec::new(),
                display,
                #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
                presenter: None,
                play,
                shown: None,
                said: CString::default(),
                pixels: Vec::new(),
                picture: (Vec::new(), (0, 0)),
            }),
        })),
        Ok(Err(message)) => set_error(message),
        Err(_) => set_error("a panic inside Orla while opening".to_owned()),
    }
}

/// An in-memory file store a session opened on it reads instead of a disk.
/// Files put after opening are read on the session's next evaluate.
pub struct OrlaFiles {
    files: std::sync::Arc<orla_session::Memory>,
}

/// A new, empty set of files.
#[unsafe(no_mangle)]
pub extern "C" fn orla_files() -> *mut OrlaFiles {
    Box::into_raw(Box::new(OrlaFiles {
        files: std::sync::Arc::default(),
    }))
}

/// Puts a file's bytes at `path`, replacing what was there.
///
/// # Safety
/// `files` is a pointer `orla_files` returned, not freed yet; `path` is a
/// NUL-terminated string; `data` is NULL when `len` is 0, or points to
/// `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_files_put(
    files: *mut OrlaFiles,
    path: *const c_char,
    data: *const u8,
    len: usize,
) -> c_int {
    let (Some(files), Some(path)) = (unsafe { files.as_ref() }, unsafe { text(path) }) else {
        return -1;
    };
    let bytes = match (data.is_null(), len) {
        (_, 0) => Vec::new(),
        (true, _) => return -1,
        (false, len) => unsafe { std::slice::from_raw_parts(data, len) }.to_vec(),
    };
    files.files.put(Path::new(path), bytes);
    0
}

/// Releases the caller's hold on a set of files; a session opened on them
/// keeps its own.
///
/// # Safety
/// `files` is NULL or a pointer `orla_files` returned, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_files_free(files: *mut OrlaFiles) {
    if !files.is_null() {
        drop(unsafe { Box::from_raw(files) });
    }
}

/// Opens a scene whose documents are `files`, as `orla_open` without the
/// disk.
///
/// # Safety
/// `files` is a pointer `orla_files` returned, not freed yet; `scene`
/// and `backend` are NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_open_files(
    files: *mut OrlaFiles,
    scene: *const c_char,
    backend: *const c_char,
) -> *mut OrlaSession {
    let Some(files) = (unsafe { files.as_ref() }) else {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string("orla_open_files needs files"));
        return std::ptr::null_mut();
    };
    let store: std::sync::Arc<dyn orla_session::Store> = files.files.clone();
    unsafe { open(Some(store), scene, backend, std::ptr::null()) }
}

/// Allocates `size` bytes in Orla's memory for a caller with no shared
/// allocator (a page, where Orla is WebAssembly); NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn orla_alloc(size: usize) -> *mut u8 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size.max(1), 8) else {
        return std::ptr::null_mut();
    };
    unsafe { std::alloc::alloc(layout) }
}

/// Frees what `orla_alloc` gave, with the size it was asked for.
///
/// # Safety
/// `pointer` is NULL or what `orla_alloc(size)` returned, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_free(pointer: *mut u8, size: usize) {
    if let (false, Ok(layout)) = (
        pointer.is_null(),
        std::alloc::Layout::from_size_align(size.max(1), 8),
    ) {
        unsafe { std::alloc::dealloc(pointer, layout) };
    }
}

/// Why the last `orla_open` on this thread failed.
#[unsafe(no_mangle)]
pub extern "C" fn orla_open_error() -> *const c_char {
    OPEN_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Ends a session.
///
/// # Safety
/// `session` is NULL or a pointer `orla_open` returned, not closed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_close(session: *mut OrlaSession) {
    // A session another call is inside is not freed under it.
    if unsafe { session.as_ref() }.is_some_and(|held| !held.inside.get()) {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// Why the last call on the session failed, or "".
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_error(session: *const OrlaSession) -> *const c_char {
    match unsafe { enter(session) } {
        Some(session) => session.error.as_ptr(),
        None => EMPTY.as_ptr(),
    }
}

/// Evaluates the scene once under `input`.
///
/// # Safety
/// `session` is a live session; `input` is NULL or points to an `OrlaInput`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_evaluate(
    session: *mut OrlaSession,
    input: *const OrlaInput,
) -> c_int {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return -1;
    };
    let input = unsafe { input.as_ref() }.copied().unwrap_or_default();
    guarded(&mut session, |s| {
        s.clear();
        s.session.sync();
        // The window: the input's, else the session's, else none.
        let (window, scale) = if input.window[0] > 0.0 && input.window[1] > 0.0 {
            (input.window, input.scale)
        } else {
            s.session.window().unwrap_or(([0.0, 0.0], 1.0))
        };
        let mut state = InputState::default();
        state.pointer = input.pointer;
        state.pointer_down = input.buttons & 1 != 0;
        state.pointer_right = input.buttons & 2 != 0;
        state.pointer_middle = input.buttons & 4 != 0;
        state.wheel = input.wheel;
        for code in 0..256 {
            if (input.keys[code >> 5] >> (code & 31)) & 1 == 1 {
                state.set_key(code as i32, true);
            }
        }
        let context = Context {
            time: input.time,
            rate: input.rate,
            input: state,
            window,
            scale,
        };
        match s.session.evaluate(&context) {
            Ok(report) => {
                s.report = c_string(&orla_protocol::report_json(&s.session, &report).to_string());
                0
            }
            Err(error) => s.fail(error.to_string()),
        }
    })
}

/// The last evaluation's report as JSON.
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_report(session: *const OrlaSession) -> *const c_char {
    match unsafe { enter(session) } {
        Some(session) => session.report.as_ptr(),
        None => EMPTY.as_ptr(),
    }
}

/// A graph output's Field.
///
/// # Safety
/// `session` is a live session; `name` is NULL or a string; `field` points
/// to an `OrlaField` to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_output(
    session: *mut OrlaSession,
    name: *const c_char,
    field: *mut OrlaField,
) -> c_int {
    let (Some(mut session), Some(field)) = (unsafe { enter(session) }, unsafe { field.as_mut() })
    else {
        return -1;
    };
    let name = unsafe { text(name) }.map(str::to_owned);
    guarded(&mut session, |s| {
        s.clear();
        let handle = match s.output(name.as_deref()) {
            Ok(handle) => handle,
            Err(message) => return s.fail(message),
        };
        match s.field(handle, field) {
            Ok(()) => 0,
            Err(message) => s.fail(message),
        }
    })
}

/// Draws a graph output (`output` NULL for the first) on the page's canvas
/// with element id `canvas`, straight from the GPU. Returns 0 when drawn,
/// 1 when there is no display or the output is no picture the device
/// holds, -1 on error.
///
/// # Safety
/// `session` is a live session; `canvas` is a string, `output` NULL or one.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_present(
    session: *mut OrlaSession,
    canvas: *const c_char,
    output: *const c_char,
) -> c_int {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return -1;
    };
    let canvas = unsafe { text(canvas) }.unwrap_or("").to_owned();
    let output = unsafe { text(output) }.map(str::to_owned);
    guarded(&mut session, |s| {
        s.clear();
        match s.present(&canvas, output.as_deref()) {
            Ok(true) => 0,
            Ok(false) => 1,
            Err(message) => s.fail(message),
        }
    })
}

/// A node's port.
///
/// # Safety
/// `session` is a live session; `node` is a string, `port` NULL or one;
/// `field` points to an `OrlaField` to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_node(
    session: *mut OrlaSession,
    node: *const c_char,
    port: *const c_char,
    field: *mut OrlaField,
) -> c_int {
    let (Some(mut session), Some(field)) = (unsafe { enter(session) }, unsafe { field.as_mut() })
    else {
        return -1;
    };
    let Some(node) = (unsafe { text(node) }).map(str::to_owned) else {
        return session.fail("orla_node needs a node name");
    };
    let port = unsafe { text(port) }.unwrap_or("result").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        let Some(handle) = s.session.cached(&orla_graph::NodeId(node.clone()), &port) else {
            return s.fail(format!(
                "node `{node}` has no materialized `{port}`; pin it and evaluate again"
            ));
        };
        match s.field(handle, field) {
            Ok(()) => 0,
            Err(message) => s.fail(message),
        }
    })
}

/// Allows the scene a directory beyond its own, as `--allow` does:
/// `file.read=<dir>` or `file.write=<dir>`. Protocol reads and writes are
/// held to the scene's directories and these.
///
/// # Safety
/// `session` is a live session; `allowance` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_allow(session: *mut OrlaSession, allowance: *const c_char) -> c_int {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return -1;
    };
    let allowance = unsafe { text(allowance) }.unwrap_or("").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        match orla_session::parse_allow(&allowance).and_then(|a| s.session.allow(&a)) {
            Ok(()) => 0,
            Err(error) => s.fail(error),
        }
    })
}

/// One protocol request line; the response line.
///
/// # Safety
/// `session` is a live session; `line` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_request(
    session: *mut OrlaSession,
    line: *const c_char,
) -> *const c_char {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return EMPTY.as_ptr();
    };
    let line = unsafe { text(line) }.unwrap_or("").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        // Its changes are recorded with origin `orla_request`, as `stdin`
        // or `socket 3` are for a served session.
        let before = s
            .session
            .act_for(orla_session::Origin::Protocol("orla_request".into()));
        let (response, _quit) = orla_protocol::respond_line(&mut s.session, &line);
        s.session.act_for(before);
        s.response = c_string(&response);
        0
    });
    session.response.as_ptr()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A call made while another is inside is refused as a NULL session's
    /// is, and the session serves again once the first leaves.
    #[test]
    fn a_call_while_another_is_inside_is_refused() {
        let scene = CString::new(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../../examples/text/scene.osn")
                .to_str()
                .unwrap(),
        )
        .unwrap();
        let cpu = CString::new("cpu").unwrap();
        let session = unsafe { orla_open(scene.as_ptr(), cpu.as_ptr(), std::ptr::null()) };
        assert!(!session.is_null());
        let status = CString::new(r#"{"op":"status"}"#).unwrap();
        {
            let _inside = unsafe { enter(session) }.expect("the first call enters");
            assert!(unsafe { enter(session) }.is_none());
            assert_eq!(unsafe { orla_evaluate(session, std::ptr::null()) }, -1);
            let answer = unsafe { CStr::from_ptr(orla_request(session, status.as_ptr())) };
            assert!(answer.to_bytes().is_empty());
            // Not freed under the call inside it.
            unsafe { orla_close(session) };
        }
        assert_eq!(unsafe { orla_evaluate(session, std::ptr::null()) }, 0);
        let answer = unsafe { CStr::from_ptr(orla_request(session, status.as_ptr())) };
        assert!(answer.to_string_lossy().contains("\"ok\""));
        unsafe { orla_close(session) };
    }
}
