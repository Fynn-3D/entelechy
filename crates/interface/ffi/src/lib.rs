//! Entelechy as a library in another process: the session the CLI and the
//! Player run, behind the C ABI in `include/entelechy.h`. A host opens a scene,
//! evaluates it under its own input, and reads Fields in place in the
//! Host's memory; the JSON protocol is reachable one line at a time.
//!
//! A session is used from one thread and one call at a time: a call made
//! while another is inside it is refused ([`EntelechySession`]). Returned
//! strings and pointers belong to the session until the next call on it;
//! a Field's words until the next evaluate, request or close. No panic
//! crosses the boundary: each entry reports one as an error.

use std::cell::{Cell, RefCell, UnsafeCell};
use std::ffi::{CStr, CString, c_char, c_int};
use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use entelechy_abi::{Context, Handle, InputState, Shape};
use entelechy_session::Session;

/// The version of the header's contract, `ENTELECHY_ABI_VERSION`.
pub const ABI_VERSION: u32 = 2;

/// What an evaluation sees; the layout of `EntelechyInput` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EntelechyInput {
    pub time: f64,
    pub rate: f32,
    pub pointer: [f32; 2],
    pub wheel: [f32; 2],
    pub window: [f32; 2],
    pub scale: f32,
    pub buttons: u32,
    pub keys: [u32; 8],
}

impl Default for EntelechyInput {
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

/// The scene's pointer as it lies in memory; the layout of
/// `EntelechyPointer` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EntelechyPointer {
    pub rgba: *const u8,
    pub width: u32,
    pub height: u32,
    pub hotspot_x: u32,
    pub hotspot_y: u32,
}

/// A Field as it lies in memory; the layout of `EntelechyField` in the header.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct EntelechyField {
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
/// call made while another is inside (a page reaching in while Entelechy waits
/// for its device) is refused as a NULL session is, so no two calls
/// touch what is held at once.
pub struct EntelechySession {
    inside: Cell<bool>,
    held: UnsafeCell<Held>,
}

/// A session entered by one call; what it holds is that call's alone
/// until it leaves.
struct Entered<'a>(&'a EntelechySession);

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
unsafe fn enter<'a>(session: *const EntelechySession) -> Option<Entered<'a>> {
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
    /// Each narrow Field read since the last evaluate or request, its
    /// words widened to the header's 32-bit words; each stays until then,
    /// as a wide Field's words in place do.
    widened: std::collections::BTreeMap<Handle, Vec<u32>>,
    /// The GPU device as a canvas reaches it, on a GPU backend.
    #[cfg_attr(
        not(all(target_arch = "wasm32", target_os = "unknown")),
        allow(dead_code)
    )]
    display: Option<entelechy_gpu::present::Display>,
    /// The canvas id last presented on, and its surface.
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    presenter: Option<(String, entelechy_gpu::present::Presenter)>,
    /// The scene played: see `play.rs`.
    play: entelechy_session::play::Play,
    /// The last frame's picture and sound, as the play names them.
    shown: Option<(Handle, Option<Handle>)>,
    /// The play's messages, for `entelechy_play_said`.
    said: CString,
    /// A frame drawn on the CPU, `0x00RRGGBB`, for `entelechy_play_pixels`.
    pixels: Vec<u32>,
    /// The last picture read back, and its size; shown again when a frame
    /// fails or a picture cannot be read back.
    picture: (Vec<[f32; 4]>, (u32, u32)),
    /// The sound heard now ([`entelechy_session::play::Play::heard`]), for
    /// `entelechy_play_sound`.
    sound: Vec<u32>,
    /// The pointer's look a frame asked for and no call has taken
    /// ([`entelechy_session::play::Play::pointer_look`]), and the picture
    /// last handed out, for `entelechy_play_pointer_look`.
    pointer_look: Option<entelechy_session::play::PointerLook>,
    pointer_rgba: Vec<u8>,
    /// The samples the host's sound device holds, as it last said
    /// (`entelechy_play_sound_queued`), for the next frame's `queued()`.
    queued: u32,
}

mod play;
pub use play::*;

/// Where a presented picture landed (`None`: the canvas took no frame)
/// and its size.
type Presented = (Option<entelechy_session::play::Placement>, (u32, u32));

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
        overlay: Option<entelechy_gpu::present::Overlay<'_>>,
    ) -> Result<Option<Presented>, String> {
        use wasm_bindgen::JsCast;
        let Some(display) = &self.display else {
            return Ok(None);
        };
        let Some((device, width, height)) = self.session.on_device(handle) else {
            return Ok(None);
        };
        // The page's canvas element; in a worker, which has no page, the
        // OffscreenCanvas the page handed over, kept by the worker under
        // the canvas's id in `entelechy_canvases`.
        let (target, window_width, window_height) = match web_sys::window()
            .and_then(|window| window.document())
        {
            Some(document) => {
                let element: web_sys::HtmlCanvasElement = document
                    .get_element_by_id(canvas)
                    .ok_or_else(|| format!("no element `{canvas}` on the page"))?
                    .dyn_into()
                    .map_err(|_| format!("`{canvas}` is not a canvas"))?;
                let (width, height) = (element.width(), element.height());
                (wgpu::SurfaceTarget::Canvas(element), width, height)
            }
            None => {
                let held = js_sys::Reflect::get(&js_sys::global(), &"entelechy_canvases".into())
                    .ok()
                    .filter(wasm_bindgen::JsValue::is_object)
                    .ok_or("a worker presents on a canvas the page handed over, kept in `entelechy_canvases`")?;
                let offscreen: web_sys::OffscreenCanvas =
                    js_sys::Reflect::get(&held, &canvas.into())
                        .ok()
                        .and_then(|value| value.dyn_into().ok())
                        .ok_or_else(|| format!("no canvas `{canvas}` in `entelechy_canvases`"))?;
                let (width, height) = (offscreen.width(), offscreen.height());
                (
                    wgpu::SurfaceTarget::OffscreenCanvas(offscreen),
                    width,
                    height,
                )
            }
        };
        if self.presenter.as_ref().is_none_or(|(id, _)| id != canvas) {
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
        _overlay: Option<entelechy_gpu::present::Overlay<'_>>,
    ) -> Result<Option<Presented>, String> {
        Err(
            "a picture is presented on a page's canvas; a program reads it with entelechy_output"
                .into(),
        )
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
    fn field(&mut self, handle: Handle, field: &mut EntelechyField) -> Result<(), String> {
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
            self.widened
                .entry(handle)
                .or_insert_with(|| data.values().into_owned())
                .as_ptr()
        };
        let values = entelechy_abi::Schema::new(schema.scalars().to_vec());
        self.schema = c_string(&values.to_string());
        *field = EntelechyField {
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
                .unwrap_or_else(|| "a panic inside Entelechy".to_owned());
            session.fail(format!("panic: {message}"))
        }
    }
}

/// The library's version.
#[unsafe(no_mangle)]
pub extern "C" fn entelechy_version() -> *const c_char {
    VERSION.as_ptr()
}

/// Opens a scene on a backend; NULL with `entelechy_open_error` set on failure.
///
/// # Safety
/// `scene`, `backend` and `library` are NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_open(
    scene: *const c_char,
    backend: *const c_char,
    library: *const c_char,
) -> *mut EntelechySession {
    unsafe { open(None, scene, backend, library) }
}

// In a page, Entelechy waits for the device by parking: `entelechy.wait` suspends
// Entelechy (JavaScript Promise Integration) until the device calls `entelechy.wake`.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "entelechy")]
unsafe extern "C" {
    fn wait();
    fn wake();
}

// A page's clock: `performance.now()`, in milliseconds.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "entelechy")]
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
fn waiting() -> entelechy_session::Wait {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        fn park() {
            unsafe { wait() }
        }
        fn woken() {
            unsafe { wake() }
        }
        entelechy_session::Wait::parks(park, woken)
    }
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    {
        entelechy_session::Wait::blocking()
    }
}

/// Opens a scene over `store`, the disk when `None`.
///
/// # Safety
/// As `entelechy_open`.
unsafe fn open(
    store: Option<std::sync::Arc<dyn entelechy_session::Store>>,
    scene: *const c_char,
    backend: *const c_char,
    library: *const c_char,
) -> *mut EntelechySession {
    let backend = unsafe { text(backend) }.unwrap_or("best");
    let library = unsafe { text(library) }.map(Path::new);
    match unsafe { text(scene) } {
        Some(scene) => opened(store, Opening::Scene(Path::new(scene), library), backend),
        None => {
            OPEN_ERROR
                .with(|slot| *slot.borrow_mut() = c_string("entelechy_open needs a scene path"));
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

/// Opens a session; NULL with `entelechy_open_error` set on failure.
fn opened(
    store: Option<std::sync::Arc<dyn entelechy_session::Store>>,
    opening: Opening<'_>,
    backend: &str,
) -> *mut EntelechySession {
    // A page's Entelechy times by the page's clock.
    if cfg!(all(target_arch = "wasm32", target_os = "unknown")) {
        entelechy_kernel::clock::give(clock);
    }
    let set_error = |message: String| {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string(&message));
        std::ptr::null_mut()
    };
    let made = catch_unwind(AssertUnwindSafe(|| {
        // The disk, finding packages where every front finds them
        // (`ENTELECHY_PACKAGES`, then a release's own); a page hands its files in.
        let store =
            store.unwrap_or_else(|| entelechy_session::SessionOptions::from_environment().store());
        let (scene, library, replay) = match opening {
            Opening::Scene(scene, library) => {
                (scene.to_path_buf(), library.map(Path::to_path_buf), None)
            }
            Opening::Replay(path) => {
                let text = store
                    .read_text(path)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                let base = path.parent().unwrap_or(Path::new(""));
                let replay = entelechy_session::replay::parse(&text, base)
                    .map_err(|e| format!("{}: {e}", path.display()))?;
                (replay.scene.clone(), replay.kernels.clone(), Some(replay))
            }
        };
        // A replay that lets its scene write keeps what it writes for the
        // play alone.
        let store = match &replay {
            Some(replay) => entelechy_session::replay::store_for(replay, store),
            None => store,
        };
        // Every session here answers `entelechy_request`, so its reads are held
        // from the start to the directories of the scene as first loaded,
        // plus what `entelechy_allow` allows.
        let limits = entelechy_session::Limits::default();
        let loaded = Session::open_loaded(
            store,
            &scene,
            library.as_deref(),
            backend,
            limits,
            waiting(),
            true,
        )?;
        let (mut session, display, passed_over) =
            (loaded.session, loaded.display, loaded.passed_over);
        // The first load publishes inline (in a page, it waits for its
        // Kernels, which prepare in turns); later replacements are prepared
        // off the caller's thread so frames go on.
        session.set_background(true);
        let mut play = entelechy_session::play::Play::new(clock);
        for passed in passed_over {
            play.say(format!("backend passed over: {passed}"));
        }
        if let Some(replay) = &replay {
            entelechy_session::replay::allow_on(replay, &mut session)?;
            play.replay(replay, &mut session);
        }
        Ok((session, display, play))
    }));
    match made {
        Ok(Ok((session, display, play))) => Box::into_raw(Box::new(EntelechySession {
            inside: Cell::new(false),
            held: UnsafeCell::new(Held {
                session,
                error: CString::default(),
                report: CString::default(),
                response: CString::default(),
                schema: CString::default(),
                widened: std::collections::BTreeMap::new(),
                display,
                #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
                presenter: None,
                play,
                shown: None,
                said: CString::default(),
                pixels: Vec::new(),
                picture: (Vec::new(), (0, 0)),
                sound: Vec::new(),
                pointer_look: None,
                pointer_rgba: Vec::new(),
                queued: 0,
            }),
        })),
        Ok(Err(message)) => set_error(message),
        Err(_) => set_error("a panic inside Entelechy while opening".to_owned()),
    }
}

/// An in-memory file store a session opened on it reads instead of a disk.
/// Files put after opening are read on the session's next evaluate.
pub struct EntelechyFiles {
    files: std::sync::Arc<entelechy_session::Memory>,
}

/// A new, empty set of files. In a page, a file put elsewhere
/// ([`entelechy_files_put_elsewhere`]) is read through `entelechy.fetch`.
#[unsafe(no_mangle)]
pub extern "C" fn entelechy_files() -> *mut EntelechyFiles {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    let files = entelechy_session::Memory::with_fetch(std::sync::Arc::new(page_fetch))
        .fetching_later(std::sync::Arc::new(page_fetch_later));
    #[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
    let files = entelechy_session::Memory::default();
    Box::into_raw(Box::new(EntelechyFiles {
        files: std::sync::Arc::new(files),
    }))
}

// In a page, a file kept by the page: `entelechy.fetch` copies up to
// `bytes` of its bytes from byte `from` on to `out` and answers how many,
// or -1.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "entelechy")]
unsafe extern "C" {
    fn fetch(path: *const u8, path_len: usize, from: usize, out: *mut u8, bytes: usize) -> isize;
}

// In a page, a file kept by the page read off the frames: `fetch_start`
// asks for it and answers a ticket; `fetch_state` answers -2 while it is
// on its way, -1 when it could not be read, else its length;
// `fetch_copy` copies up to `bytes` of it from byte `from` on to `out` and
// answers how many; `fetch_end` lets it go.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "entelechy")]
unsafe extern "C" {
    fn fetch_start(path: *const u8, path_len: usize) -> u32;
    fn fetch_state(ticket: u32) -> isize;
    fn fetch_copy(ticket: u32, from: usize, out: *mut u8, bytes: usize) -> isize;
    fn fetch_end(ticket: u32);
}

/// A file the page keeps, on its way ([`entelechy_session::Later`]).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct PageLater(Option<u32>);

/// A file the page keeps, in, held by the page until let go
/// ([`entelechy_session::Arrived`]).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
struct PageArrived {
    ticket: u32,
    len: usize,
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl entelechy_session::Later for PageLater {
    fn poll(&mut self) -> Option<std::io::Result<Box<dyn entelechy_session::Arrived>>> {
        let ticket = self.0?;
        let state = unsafe { fetch_state(ticket) };
        if state == -2 {
            return None;
        }
        self.0 = None;
        if state < 0 {
            unsafe { fetch_end(ticket) };
            return Some(Err(std::io::ErrorKind::NotFound.into()));
        }
        Some(Ok(Box::new(PageArrived {
            ticket,
            len: state as usize,
        })))
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Drop for PageLater {
    fn drop(&mut self) {
        if let Some(ticket) = self.0 {
            unsafe { fetch_end(ticket) };
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl entelechy_session::Arrived for PageArrived {
    fn len(&self) -> usize {
        self.len
    }

    fn start(&self, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes.min(self.len)];
        let got = unsafe { fetch_copy(self.ticket, 0, out.as_mut_ptr(), out.len()) };
        out.truncate(got.max(0) as usize);
        out
    }

    fn words(&self, from: usize) -> (Vec<u32>, usize) {
        // Copied straight into the words: little-endian, as a page is.
        let bytes = self.len.saturating_sub(from);
        let mut words = vec![0u32; bytes.div_ceil(4)];
        let got = unsafe { fetch_copy(self.ticket, from, words.as_mut_ptr().cast::<u8>(), bytes) };
        (words, got.max(0) as usize)
    }
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
impl Drop for PageArrived {
    fn drop(&mut self) {
        unsafe { fetch_end(self.ticket) };
    }
}

/// A file the page keeps, asked for off the frames
/// ([`entelechy_session::FetchLaterFn`]).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn page_fetch_later(path: &Path) -> Box<dyn entelechy_session::Later> {
    let name = path.to_string_lossy();
    Box::new(PageLater(Some(unsafe {
        fetch_start(name.as_ptr(), name.len())
    })))
}

/// A file the page keeps, from byte `from` on into `out`
/// ([`entelechy_session::FetchFn`]).
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn page_fetch(path: &Path, from: usize, out: &mut [u8]) -> std::io::Result<usize> {
    let name = path.to_string_lossy();
    let got = unsafe { fetch(name.as_ptr(), name.len(), from, out.as_mut_ptr(), out.len()) };
    if got < 0 {
        return Err(std::io::Error::other(format!(
            "the page could not fetch `{name}`"
        )));
    }
    Ok(got as usize)
}

/// Puts a file of `len` bytes at `path` whose bytes the caller keeps: in a
/// page, read through `entelechy.fetch` whenever the file is read, so they
/// are not held twice. Elsewhere such a file cannot be read.
///
/// # Safety
/// `files` is a pointer `entelechy_files` returned, not freed yet; `path` is a
/// NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_files_put_elsewhere(
    files: *mut EntelechyFiles,
    path: *const c_char,
    len: usize,
) -> c_int {
    let (Some(files), Some(path)) = (unsafe { files.as_ref() }, unsafe { text(path) }) else {
        return -1;
    };
    files.files.put_elsewhere(Path::new(path), len);
    0
}

/// Puts a file's bytes at `path`, replacing what was there.
///
/// # Safety
/// `files` is a pointer `entelechy_files` returned, not freed yet; `path` is a
/// NUL-terminated string; `data` is NULL when `len` is 0, or points to
/// `len` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_files_put(
    files: *mut EntelechyFiles,
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
/// `files` is NULL or a pointer `entelechy_files` returned, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_files_free(files: *mut EntelechyFiles) {
    if !files.is_null() {
        drop(unsafe { Box::from_raw(files) });
    }
}

/// Opens a scene whose documents are `files`, as `entelechy_open` without the
/// disk.
///
/// # Safety
/// `files` is a pointer `entelechy_files` returned, not freed yet; `scene`
/// and `backend` are NULL or NUL-terminated strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_open_files(
    files: *mut EntelechyFiles,
    scene: *const c_char,
    backend: *const c_char,
) -> *mut EntelechySession {
    let Some(files) = (unsafe { files.as_ref() }) else {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string("entelechy_open_files needs files"));
        return std::ptr::null_mut();
    };
    let store: std::sync::Arc<dyn entelechy_session::Store> = files.files.clone();
    unsafe { open(Some(store), scene, backend, std::ptr::null()) }
}

/// Allocates `size` bytes in Entelechy's memory for a caller with no shared
/// allocator (a page, where Entelechy is WebAssembly); NULL on failure.
#[unsafe(no_mangle)]
pub extern "C" fn entelechy_alloc(size: usize) -> *mut u8 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size.max(1), 8) else {
        return std::ptr::null_mut();
    };
    unsafe { std::alloc::alloc(layout) }
}

/// Frees what `entelechy_alloc` gave, with the size it was asked for.
///
/// # Safety
/// `pointer` is NULL or what `entelechy_alloc(size)` returned, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_free(pointer: *mut u8, size: usize) {
    if let (false, Ok(layout)) = (
        pointer.is_null(),
        std::alloc::Layout::from_size_align(size.max(1), 8),
    ) {
        unsafe { std::alloc::dealloc(pointer, layout) };
    }
}

/// Why the last `entelechy_open` on this thread failed.
#[unsafe(no_mangle)]
pub extern "C" fn entelechy_open_error() -> *const c_char {
    OPEN_ERROR.with(|slot| slot.borrow().as_ptr())
}

/// Ends a session, the frame in flight confirmed first, so what it
/// writes to files is written.
///
/// # Safety
/// `session` is NULL or a pointer `entelechy_open` returned, not closed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_close(session: *mut EntelechySession) {
    // A session another call is inside is not freed under it.
    if unsafe { session.as_ref() }.is_some_and(|held| !held.inside.get()) {
        let mut closed = unsafe { Box::from_raw(session) };
        let _ = closed.held.get_mut().session.confirm();
    }
}

/// Why the last call on the session failed, or "".
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_error(session: *const EntelechySession) -> *const c_char {
    match unsafe { enter(session) } {
        Some(session) => session.error.as_ptr(),
        None => EMPTY.as_ptr(),
    }
}

/// Evaluates the scene once under `input`.
///
/// # Safety
/// `session` is a live session; `input` is NULL or points to an `EntelechyInput`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_evaluate(
    session: *mut EntelechySession,
    input: *const EntelechyInput,
) -> c_int {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return -1;
    };
    let input = unsafe { input.as_ref() }.copied().unwrap_or_default();
    guarded(&mut session, |s| {
        s.clear();
        s.widened.clear();
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
        // A host evaluating on its own keeps no sound device's queue.
        let context = Context {
            time: input.time,
            rate: input.rate,
            queued: 0,
            input: state,
            window,
            scale,
        };
        match s.session.evaluate(&context) {
            Ok(report) => {
                s.report =
                    c_string(&entelechy_protocol::report_json(&s.session, &report).to_string());
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
pub unsafe extern "C" fn entelechy_report(session: *const EntelechySession) -> *const c_char {
    match unsafe { enter(session) } {
        Some(session) => session.report.as_ptr(),
        None => EMPTY.as_ptr(),
    }
}

/// A graph output's Field.
///
/// # Safety
/// `session` is a live session; `name` is NULL or a string; `field` points
/// to an `EntelechyField` to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_output(
    session: *mut EntelechySession,
    name: *const c_char,
    field: *mut EntelechyField,
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
pub unsafe extern "C" fn entelechy_present(
    session: *mut EntelechySession,
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
/// `field` points to an `EntelechyField` to fill.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn entelechy_node(
    session: *mut EntelechySession,
    node: *const c_char,
    port: *const c_char,
    field: *mut EntelechyField,
) -> c_int {
    let (Some(mut session), Some(field)) = (unsafe { enter(session) }, unsafe { field.as_mut() })
    else {
        return -1;
    };
    let Some(node) = (unsafe { text(node) }).map(str::to_owned) else {
        return session.fail("entelechy_node needs a node name");
    };
    let port = unsafe { text(port) }.unwrap_or("result").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        let id = entelechy_graph::NodeId(node.clone().into());
        // Outputs let go cold are made again to be read.
        if let Err(error) = s.session.warm(&id, &port) {
            return s.fail(error.to_string());
        }
        let Some(handle) = s.session.cached(&id, &port) else {
            let why = s.session.why_no_value(&id, &port);
            return s.fail(why);
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
pub unsafe extern "C" fn entelechy_allow(
    session: *mut EntelechySession,
    allowance: *const c_char,
) -> c_int {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return -1;
    };
    let allowance = unsafe { text(allowance) }.unwrap_or("").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        match entelechy_session::parse_allow(&allowance).and_then(|a| s.session.allow(&a)) {
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
pub unsafe extern "C" fn entelechy_request(
    session: *mut EntelechySession,
    line: *const c_char,
) -> *const c_char {
    let Some(mut session) = (unsafe { enter(session) }) else {
        return EMPTY.as_ptr();
    };
    let line = unsafe { text(line) }.unwrap_or("").to_owned();
    guarded(&mut session, |s| {
        s.clear();
        s.widened.clear();
        // Its changes are recorded with origin `entelechy_request`, as `stdin`
        // or `socket 3` are for a served session.
        let before = s.session.act_for(entelechy_session::Origin::Protocol(
            "entelechy_request".into(),
        ));
        let (response, _quit) = entelechy_protocol::respond_line(&mut s.session, &line);
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
        let dir = std::env::temp_dir().join(format!("entelechy-ffi-inside-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("scene.esn"),
            "node fill: kernel { requires { field.write; } output result: Image; domain = vec2(4, 4); compute { return vec4(0.5, 0.5, 0.5, 1.0); } }\noutput result <- fill.result;\n",
        )
        .unwrap();
        let scene = CString::new(dir.join("scene.esn").to_str().unwrap()).unwrap();
        let cpu = CString::new("cpu").unwrap();
        let session = unsafe { entelechy_open(scene.as_ptr(), cpu.as_ptr(), std::ptr::null()) };
        assert!(!session.is_null());
        let status = CString::new(r#"{"op":"status"}"#).unwrap();
        {
            let _inside = unsafe { enter(session) }.expect("the first call enters");
            assert!(unsafe { enter(session) }.is_none());
            assert_eq!(unsafe { entelechy_evaluate(session, std::ptr::null()) }, -1);
            let answer = unsafe { CStr::from_ptr(entelechy_request(session, status.as_ptr())) };
            assert!(answer.to_bytes().is_empty());
            // Not freed under the call inside it.
            unsafe { entelechy_close(session) };
        }
        assert_eq!(unsafe { entelechy_evaluate(session, std::ptr::null()) }, 0);
        let answer = unsafe { CStr::from_ptr(entelechy_request(session, status.as_ptr())) };
        assert!(answer.to_string_lossy().contains("\"ok\""));
        unsafe { entelechy_close(session) };
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A narrow Field's widened words stay until the next evaluate, as
    /// the header says, however many other Fields are read meanwhile.
    #[test]
    fn widened_words_outlive_other_reads() {
        let dir =
            std::env::temp_dir().join(format!("entelechy-ffi-widened-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let node = |name: &str, value: i32| {
            format!(
                "node {name}: kernel {{ requires {{ field.write; }} output result: Rows<Int as u8>; domain = 4096; compute {{ return {value}; }} }}\n"
            )
        };
        let text = format!(
            "{}{}output a <- a.result;\noutput b <- b.result;\n",
            node("a", 1),
            node("b", 2)
        );
        std::fs::write(dir.join("scene.esn"), text).unwrap();
        let scene = CString::new(dir.join("scene.esn").to_str().unwrap()).unwrap();
        let cpu = CString::new("cpu").unwrap();
        let session = unsafe { entelechy_open(scene.as_ptr(), cpu.as_ptr(), std::ptr::null()) };
        assert!(!session.is_null());
        assert_eq!(unsafe { entelechy_evaluate(session, std::ptr::null()) }, 0);
        let empty = || EntelechyField {
            words: std::ptr::null(),
            width: 0,
            height: 0,
            depth: 0,
            count: 0,
            stride: 0,
            shape: 0,
            pixels: 0,
            schema: std::ptr::null(),
        };
        let (name_a, name_b) = (CString::new("a").unwrap(), CString::new("b").unwrap());
        let mut a = empty();
        assert_eq!(
            unsafe { entelechy_output(session, name_a.as_ptr(), &mut a) },
            0
        );
        for _ in 0..4 {
            let mut b = empty();
            assert_eq!(
                unsafe { entelechy_output(session, name_b.as_ptr(), &mut b) },
                0
            );
        }
        // Read through the pointer the first call gave.
        let words = unsafe { std::slice::from_raw_parts(a.words, a.count as usize) };
        assert!(words.iter().all(|word| *word == 1), "{:?}", &words[..8]);
        unsafe { entelechy_close(session) };
        let _ = std::fs::remove_dir_all(&dir);
    }
}
