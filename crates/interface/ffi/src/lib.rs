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
    let set_error = |message: String| {
        OPEN_ERROR.with(|slot| *slot.borrow_mut() = c_string(&message));
        std::ptr::null_mut()
    };
    let opened = catch_unwind(|| {
        let Some(scene) = (unsafe { text(scene) }) else {
            return Err("orla_open needs a scene path".to_owned());
        };
        let backend = unsafe { text(backend) }.unwrap_or("jit");
        let library = unsafe { text(library) }.map(Path::new);
        let runtime = Session::runtime_for(backend, orla_session::Limits::default())?;
        let mut session = Session::open_with(Path::new(scene), library, runtime);
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
    });
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
