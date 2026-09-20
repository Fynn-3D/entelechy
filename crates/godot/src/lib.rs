//! Orla inside Godot's process: a GDExtension whose one class,
//! `OrlaSession`, wraps the C ABI of `orla-ffi`. A scene opens on a
//! backend, evaluates under an input the script assembles, and its
//! Fields come out as an `Image` (RGBAF, one copy from the Host's words
//! into Godot's buffer, no socket and no encoding) or as words with
//! their layout; one protocol line in and one out for everything else.
//! Nothing here knows how Orla is built beyond the header's contract.

use std::ffi::{CStr, CString};

use godot::classes::image::Format;
use godot::classes::{Image, ProjectSettings};
use godot::prelude::*;
use orla_ffi::{OrlaField, OrlaInput};

struct OrlaExtension;

#[gdextension]
unsafe impl ExtensionLibrary for OrlaExtension {}

/// A running scene, in this process.
#[derive(GodotClass)]
#[class(base = RefCounted)]
pub struct OrlaSession {
    session: *mut orla_ffi::OrlaSession,
    error: String,
    base: Base<RefCounted>,
}

#[godot_api]
impl IRefCounted for OrlaSession {
    fn init(base: Base<RefCounted>) -> Self {
        Self {
            session: std::ptr::null_mut(),
            error: String::new(),
            base,
        }
    }
}

impl Drop for OrlaSession {
    fn drop(&mut self) {
        self.close();
    }
}

fn c_string(text: &str) -> CString {
    CString::new(text.replace('\0', " ")).unwrap_or_default()
}

fn string_at(pointer: *const std::ffi::c_char) -> String {
    if pointer.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(pointer) }
            .to_string_lossy()
            .into_owned()
    }
}

fn empty_field() -> OrlaField {
    OrlaField {
        words: std::ptr::null(),
        width: 0,
        height: 0,
        depth: 0,
        count: 0,
        stride: 0,
        shape: 0,
        pixels: 0,
        schema: std::ptr::null(),
    }
}

/// The bytes of a Field's words, copied once.
fn bytes_of(field: &OrlaField) -> PackedByteArray {
    let words = (field.count as usize) * (field.stride as usize);
    if field.words.is_null() || words == 0 {
        return PackedByteArray::new();
    }
    let slice = unsafe { std::slice::from_raw_parts(field.words.cast::<u8>(), words * 4) };
    PackedByteArray::from(slice)
}

/// A Field as a VarDictionary: its words as bytes and its layout.
fn dictionary_of(field: &OrlaField) -> VarDictionary {
    let mut dictionary = VarDictionary::new();
    dictionary.set("words", bytes_of(field));
    dictionary.set("width", field.width as i64);
    dictionary.set("height", field.height as i64);
    dictionary.set("depth", field.depth as i64);
    dictionary.set("count", field.count as i64);
    dictionary.set("stride", field.stride as i64);
    dictionary.set(
        "shape",
        match field.shape {
            0 => "Rows",
            1 => "Grid",
            _ => "Volume",
        },
    );
    dictionary.set("pixels", field.pixels == 1);
    dictionary.set("schema", string_at(field.schema));
    dictionary
}

#[godot_api]
impl OrlaSession {
    /// Opens a scene (a `res://` or file path to a `.osn`) on a backend
    /// (`"cpu"`, `"gpu"` or `"jit"`, `""` for cpu); false with `error()`
    /// set when it does not open. An open session is closed first.
    #[func]
    pub fn open(&mut self, scene: GString, backend: GString) -> bool {
        self.close();
        let path = ProjectSettings::singleton().globalize_path(&scene);
        let scene = c_string(&path.to_string());
        let backend = backend.to_string();
        let backend_c = c_string(if backend.is_empty() { "cpu" } else { &backend });
        let session =
            unsafe { orla_ffi::orla_open(scene.as_ptr(), backend_c.as_ptr(), std::ptr::null()) };
        if session.is_null() {
            self.error = string_at(orla_ffi::orla_open_error());
            return false;
        }
        self.session = session;
        self.error.clear();
        true
    }

    /// Whether a scene is open.
    #[func]
    pub fn is_open(&self) -> bool {
        !self.session.is_null()
    }

    /// Closes the scene, if one is open.
    #[func]
    pub fn close(&mut self) {
        if !self.session.is_null() {
            unsafe { orla_ffi::orla_close(self.session) };
            self.session = std::ptr::null_mut();
        }
    }

    /// Why the last call failed, or "".
    #[func]
    pub fn error(&self) -> GString {
        GString::from(&self.error)
    }

    fn take_error(&mut self) {
        self.error = string_at(unsafe { orla_ffi::orla_error(self.session) });
    }

    /// Evaluates the scene once. `input` may hold `time`, `rate`,
    /// `pointer` (Vector2, in output pixels), `wheel` (Vector2),
    /// `window` (Vector2 or Vector2i, physical pixels; zero for the
    /// session's own), `scale`, `buttons` (an int of bits or an Array of
    /// numbers, 0 left, 1 right, 2 middle) and `keys` (an Array of codes).
    /// False with `error()` set when the evaluation fails.
    #[func]
    pub fn evaluate(&mut self, input: VarDictionary) -> bool {
        if self.session.is_null() {
            self.error = "no scene is open".into();
            return false;
        }
        let mut native = OrlaInput::default();
        // A number in a script's Dictionary may be a float or an int.
        let as_f32 = |value: &Variant| -> Option<f32> {
            value
                .try_to::<f64>()
                .ok()
                .map(|v| v as f32)
                .or_else(|| value.try_to::<i64>().ok().map(|v| v as f32))
        };
        let number = |key: &str| -> Option<f32> { input.get(key).and_then(|v| as_f32(&v)) };
        let pair = |key: &str| -> Option<[f32; 2]> {
            let value = input.get(key)?;
            if let Ok(v) = value.try_to::<Vector2>() {
                return Some([v.x, v.y]);
            }
            if let Ok(v) = value.try_to::<Vector2i>() {
                return Some([v.x as f32, v.y as f32]);
            }
            let array = value.try_to::<VarArray>().ok()?;
            let at = |i: usize| {
                array
                    .get(i)
                    .and_then(|v| as_f32(&v).map(f64::from))
                    .unwrap_or(0.0) as f32
            };
            Some([at(0), at(1)])
        };
        native.time = number("time").unwrap_or(0.0);
        native.rate = number("rate").unwrap_or(0.0);
        native.scale = number("scale").unwrap_or(1.0);
        if let Some(p) = pair("pointer") {
            native.pointer = p;
        }
        if let Some(w) = pair("wheel") {
            native.wheel = w;
        }
        if let Some(w) = pair("window") {
            native.window = w;
        }
        if let Some(buttons) = input.get("buttons") {
            if let Ok(bits) = buttons.try_to::<i64>() {
                native.buttons = bits as u32;
            } else if let Ok(list) = buttons.try_to::<VarArray>() {
                for item in list.iter_shared() {
                    if let Some(button) = as_f32(&item).map(|b| b as i64)
                        && (0..3).contains(&button)
                    {
                        native.buttons |= 1 << button;
                    }
                }
            }
        }
        if let Some(keys) = input.get("keys")
            && let Ok(list) = keys.try_to::<VarArray>()
        {
            for item in list.iter_shared() {
                if let Some(code) = as_f32(&item).map(|c| c as i64)
                    && (0..256).contains(&code)
                {
                    native.keys[(code >> 5) as usize] |= 1 << (code & 31);
                }
            }
        }
        let ok = unsafe { orla_ffi::orla_evaluate(self.session, &native) } == 0;
        self.take_error();
        ok
    }

    /// The last evaluation's report as JSON text.
    #[func]
    pub fn report(&self) -> GString {
        if self.session.is_null() {
            return GString::new();
        }
        GString::from(&string_at(unsafe { orla_ffi::orla_report(self.session) }))
    }

    /// A graph output's Field (`name` "" for the first) as a VarDictionary:
    /// `words` (PackedByteArray of 32-bit words, little-endian), `width`,
    /// `height`, `depth`, `count`, `stride`, `shape`, `pixels`, `schema`.
    /// Empty with `error()` set on failure.
    #[func]
    pub fn output(&mut self, name: GString) -> VarDictionary {
        if self.session.is_null() {
            self.error = "no scene is open".into();
            return VarDictionary::new();
        }
        let name = name.to_string();
        let name_c = c_string(&name);
        let mut field = empty_field();
        let pointer = if name.is_empty() {
            std::ptr::null()
        } else {
            name_c.as_ptr()
        };
        let ok = unsafe { orla_ffi::orla_output(self.session, pointer, &mut field) } == 0;
        self.take_error();
        if ok {
            dictionary_of(&field)
        } else {
            VarDictionary::new()
        }
    }

    /// A graph output that is a picture, as an RGBAF `Image` (`name` ""
    /// for the first output); null with `error()` set otherwise.
    #[func]
    pub fn output_image(&mut self, name: GString) -> Option<Gd<Image>> {
        if self.session.is_null() {
            self.error = "no scene is open".into();
            return None;
        }
        let name = name.to_string();
        let name_c = c_string(&name);
        let mut field = empty_field();
        let pointer = if name.is_empty() {
            std::ptr::null()
        } else {
            name_c.as_ptr()
        };
        let ok = unsafe { orla_ffi::orla_output(self.session, pointer, &mut field) } == 0;
        self.take_error();
        if !ok {
            return None;
        }
        if field.pixels != 1 || field.stride != 4 {
            self.error = format!(
                "output `{name}` is not a picture ({})",
                string_at(field.schema)
            );
            return None;
        }
        Image::create_from_data(
            field.width as i32,
            field.height as i32,
            false,
            Format::RGBAF,
            &bytes_of(&field),
        )
    }

    /// A node's port as a VarDictionary like `output`; a node that fusion
    /// would inline is pinned first with `request('{"op":"pin",...}')`.
    #[func]
    pub fn node(&mut self, node: GString, port: GString) -> VarDictionary {
        if self.session.is_null() {
            self.error = "no scene is open".into();
            return VarDictionary::new();
        }
        let node_c = c_string(&node.to_string());
        let port = port.to_string();
        let port_c = c_string(&port);
        let mut field = empty_field();
        let port_pointer = if port.is_empty() {
            std::ptr::null()
        } else {
            port_c.as_ptr()
        };
        let ok =
            unsafe { orla_ffi::orla_node(self.session, node_c.as_ptr(), port_pointer, &mut field) }
                == 0;
        self.take_error();
        if ok {
            dictionary_of(&field)
        } else {
            VarDictionary::new()
        }
    }

    /// One line of the creation protocol; its response line.
    #[func]
    pub fn request(&mut self, line: GString) -> GString {
        if self.session.is_null() {
            self.error = "no scene is open".into();
            return GString::new();
        }
        let line_c = c_string(&line.to_string());
        let response = string_at(unsafe { orla_ffi::orla_request(self.session, line_c.as_ptr()) });
        self.take_error();
        GString::from(&response)
    }
}
