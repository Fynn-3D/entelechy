//! Orla as a library in another program's process: the session the
//! command line and the Player run, behind a C ABI declared in
//! `include/orla.h`. A host opens a scene, evaluates it under its own
//! input, and reads the Fields where they lie in the Host's memory, no
//! socket and no encoding between; the JSON protocol is reachable too,
//! one line at a time, for everything the fast path does not name.
//!
//! Every function takes one session from one thread at a time. Strings
//! and pointers returned belong to the session until the next call on
//! it; a Field's words until the next evaluate, request or close. No
//! panic crosses the boundary: each entry catches one and reports it as
//! an error.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_int};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use orla_abi::{Context, Handle, InputState, Shape};
use orla_session::{Event, Session};

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

/// A running scene and the buffers its answers live in.
pub struct OrlaSession {
    session: Session,
    error: CString,
    report: CString,
    response: CString,
    schema: CString,
    /// A narrow Field's words widened, which a caller reads instead of
    /// the packed memory: the header promises 32-bit words.
    widened: Vec<u32>,
}

thread_local! {
    static OPEN_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

const VERSION: &CStr = c"0.1.0";
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

impl OrlaSession {
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
        self.session
            .runtime_mut()
            .materialize(handle)
            .map_err(|error| error.to_string())?;
        let host = self.session.runtime().host();
        let data = host.data(handle).map_err(|error| error.to_string())?;
        let domain = data.domain();
        let schema = data.schema();
        // Whole words are read where they lie; a Field kept narrower is
        // widened into the session's own buffer, so the header's promise
        // (32-bit words, a layout of `f`, `i` and `b`) holds either way.
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
fn guarded(session: &mut OrlaSession, body: impl FnOnce(&mut OrlaSession) -> c_int) -> c_int {
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

// In a page, Orla waits for the device by giving the page a turn: the
// page's `orla.wait` suspends Orla (JavaScript Promise Integration) until
// the browser has had one.
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[link(wasm_import_module = "orla")]
unsafe extern "C" {
    fn wait();
}

/// How this process waits for a device.
fn waiting() -> orla_session::Wait {
    #[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
    {
        fn turn() {
            unsafe { wait() }
        }
        orla_session::Wait::turns(turn)
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
    let set_error = |message: String| {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string(&message));
        std::ptr::null_mut()
    };
    let opened = catch_unwind(AssertUnwindSafe(|| {
        let Some(scene) = (unsafe { text(scene) }) else {
            return Err("orla_open needs a scene path".to_owned());
        };
        let backend = unsafe { text(backend) }.unwrap_or("jit");
        let library = unsafe { text(library) }.map(Path::new);
        let runtime =
            Session::runtime_waiting(backend, orla_session::Limits::default(), waiting())?;
        let store = store.unwrap_or_else(|| std::sync::Arc::new(orla_session::Disk));
        let mut session = Session::open_in(store, Path::new(scene), library, runtime);
        let events = session.sync();
        if session.graph().is_none() {
            let reason = events
                .iter()
                .find_map(|event| match event {
                    Event::SceneRejected(error) => Some(error.to_string()),
                    Event::Rejected { path, error } => Some(format!("{}: {error}", path.display())),
                    _ => None,
                })
                .unwrap_or_else(|| format!("{scene} did not load"));
            return Err(reason);
        }
        Ok(session)
    }));
    match opened {
        Ok(Ok(session)) => Box::into_raw(Box::new(OrlaSession {
            session,
            error: CString::default(),
            report: CString::default(),
            response: CString::default(),
            schema: CString::default(),
            widened: Vec::new(),
        })),
        Ok(Err(message)) => set_error(message),
        Err(_) => set_error("a panic inside Orla while opening".to_owned()),
    }
}

/// Files handed in: a store in memory that a session opened on it reads
/// instead of a disk. The caller keeps it and may put files after the
/// session opened; the session reads them on its next evaluate.
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

/// Lets go of the caller's hold on a set of files; a session opened on
/// them keeps its own.
///
/// # Safety
/// `files` is NULL or a pointer `orla_files` returned, not freed yet.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_files_free(files: *mut OrlaFiles) {
    if !files.is_null() {
        drop(unsafe { Box::from_raw(files) });
    }
}

/// Opens a scene whose documents are `files`: as `orla_open`, with no
/// disk read.
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

/// Room for `size` bytes in Orla's memory, for a caller that shares no
/// allocator with it (a page, where Orla is WebAssembly) to write a
/// string or a file into; NULL when there is none.
#[unsafe(no_mangle)]
pub extern "C" fn orla_alloc(size: usize) -> *mut u8 {
    let Ok(layout) = std::alloc::Layout::from_size_align(size.max(1), 8) else {
        return std::ptr::null_mut();
    };
    unsafe { std::alloc::alloc(layout) }
}

/// Gives back what `orla_alloc` gave, with the size it was asked for.
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
    if !session.is_null() {
        drop(unsafe { Box::from_raw(session) });
    }
}

/// Why the last call on the session failed, or "".
///
/// # Safety
/// `session` is NULL or a live session.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_error(session: *const OrlaSession) -> *const c_char {
    match unsafe { session.as_ref() } {
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
    let Some(session) = (unsafe { session.as_mut() }) else {
        return -1;
    };
    let input = unsafe { input.as_ref() }.copied().unwrap_or_default();
    guarded(session, |s| {
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
    match unsafe { session.as_ref() } {
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
    let (Some(session), Some(field)) = (unsafe { session.as_mut() }, unsafe { field.as_mut() })
    else {
        return -1;
    };
    let name = unsafe { text(name) }.map(str::to_owned);
    guarded(session, |s| {
        s.clear();
        let Some(graph) = s.session.graph() else {
            return s.fail("no scene is loaded");
        };
        let found = match &name {
            Some(name) => graph.outputs.iter().find(|(output, _)| output == name),
            None => graph.outputs.first(),
        };
        let Some((name, source)) = found else {
            return s.fail(match &name {
                Some(name) => format!("no output named `{name}`"),
                None => "the graph has no output".to_owned(),
            });
        };
        let handle = match source {
            orla_graph::Source::Node { node, port } => s.session.cached(node, port),
            orla_graph::Source::External(_) => {
                return s.fail(format!("output `{name}` is an external image"));
            }
        };
        let Some(handle) = handle else {
            return s.fail("evaluate before reading an output");
        };
        match s.field(handle, field) {
            Ok(()) => 0,
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
    let (Some(session), Some(field)) = (unsafe { session.as_mut() }, unsafe { field.as_mut() })
    else {
        return -1;
    };
    let Some(node) = (unsafe { text(node) }).map(str::to_owned) else {
        return session.fail("orla_node needs a node name");
    };
    let port = unsafe { text(port) }.unwrap_or("result").to_owned();
    guarded(session, |s| {
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

/// One protocol request line; the response line.
///
/// # Safety
/// `session` is a live session; `line` is a NUL-terminated string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn orla_request(
    session: *mut OrlaSession,
    line: *const c_char,
) -> *const c_char {
    let Some(session) = (unsafe { session.as_mut() }) else {
        return EMPTY.as_ptr();
    };
    let line = unsafe { text(line) }.unwrap_or("").to_owned();
    guarded(session, |s| {
        s.clear();
        let (response, _quit) = orla_protocol::respond_line(&mut s.session, &line);
        s.response = c_string(&response);
        0
    });
    session.response.as_ptr()
}
