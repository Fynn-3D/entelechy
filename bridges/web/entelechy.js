// Entelechy's creation protocol from a browser. A client of the public
// contracts and nothing more: JSON lines as WebSocket text frames to a
// session serving with `--listen`, the same lines Blender or a shell
// would send on the socket; or the same lines answered in the page by
// Entelechy built as WebAssembly, through its C ABI.
//
// A classic script, not a module, so a page opened from a file works:
// browsers refuse module imports between file URLs. It defines `Entelechy`.

(() => {
/// A connection. The session answers one line per request, in order, so
/// answers are matched to requests first in, first out; the `id` a
/// request carries is echoed but not relied on, since `kernel` and
/// `retire` use that field for a Kernel's id.
class EntelechyClient {
  constructor(socket) {
    this.socket = socket;
    this.pending = [];
    this.onclose = null;
    socket.onmessage = (event) => {
      let response;
      try {
        response = JSON.parse(event.data);
      } catch (error) {
        return;
      }
      const waiting = this.pending.shift();
      if (!waiting) return;
      if ("error" in response) {
        const detail = response.error;
        const error = new Error(detail.message);
        error.line = detail.line;
        error.column = detail.column;
        waiting.reject(error);
      } else {
        waiting.resolve(response.ok);
      }
    };
    socket.onclose = () => {
      for (const waiting of this.pending) {
        waiting.reject(new Error("the connection closed"));
      }
      this.pending = [];
      if (this.onclose) this.onclose();
    };
  }

  /// Opens `url` (`ws://127.0.0.1:7788`) and resolves once the session answers.
  static open(url) {
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(url);
      socket.onopen = () => resolve(new EntelechyClient(socket));
      socket.onerror = () => reject(new Error(`cannot connect to ${url}`));
    });
  }

  /// One request; resolves with the `ok` payload, rejects with the error
  /// (its `line` and `column` set when the session gave a position).
  call(op, fields = {}) {
    const line = JSON.stringify({ op, ...fields });
    return new Promise((resolve, reject) => {
      const waiting = { resolve, reject };
      this.pending.push(waiting);
      try {
        this.socket.send(line);
      } catch (error) {
        this.pending = this.pending.filter((entry) => entry !== waiting);
        reject(error);
      }
    });
  }

  close() {
    this.socket.close();
  }
}

/// Parking for Entelechy while it waits for the device: `park` resolves once
/// `wake` has been called since it last resolved, which the device's
/// answer does; a second, at most, when nothing wakes it, so a wait the
/// device never answers is not a page stopped for good.
function parking() {
  let woken = false;
  let waiting = null;
  return {
    park() {
      if (woken) {
        woken = false;
        return Promise.resolve();
      }
      return new Promise((done) => {
        let timer = 0;
        const resume = () => {
          if (waiting !== resume) return;
          waiting = null;
          woken = false;
          if (timer) clearTimeout(timer);
          done();
        };
        waiting = resume;
        timer = setTimeout(resume, 1000);
      });
    },
    wake() {
      woken = true;
      if (waiting) waiting();
    },
  };
}

/// What Entelechy built as WebAssembly imports from its host: `entelechy.load`,
/// which compiles a Kernel's code (WebAssembly the JIT wrote) over Entelechy's
/// memory and function table and puts its functions into the table;
/// `entelechy.unload`, which gives their slots back; `entelechy.wait`, which
/// suspends Entelechy (JavaScript Promise Integration) until `entelechy.wake`,
/// which the device's answer calls, which is how Entelechy waits for a
/// device; `entelechy.now`, the page's clock in milliseconds; and
/// `entelechy.fetch`, which reads a file the page keeps (put elsewhere),
/// from a byte on, from where `globalThis.entelechy_files_base` says the
/// files are; and `entelechy.fetch_start`, `fetch_state`, `fetch_copy` and
/// `fetch_end`, which read one off the frames, as the browser fetches it.
/// `instance()` answers the Entelechy instance, once there is one.
function host(instance) {
  const free = [];
  const parked = parking();
  // The file `entelechy.fetch` last read, while reads go on through it.
  let fetched = null;
  // Files asked for off the frames (`entelechy.fetch_start`), by ticket:
  // -2 while on their way, -1 when not read, else their bytes.
  const coming = new Map();
  let tickets = 0;
  const table = () => instance().exports.__indirect_function_table;
  return {
    load(bytes, length, frame, slots, most) {
      if (!length) return typeof WebAssembly.Module === "function" ? 0 : -2;
      const exports = instance().exports;
      try {
        const code = new Uint8Array(exports.memory.buffer, bytes, length).slice();
        const loaded = new WebAssembly.Instance(new WebAssembly.Module(code), {
          env: { memory: exports.memory, table: table() },
        });
        if (loaded.exports.frame) loaded.exports.frame.value = frame | 0;
        const out = new Uint32Array(exports.memory.buffer, slots, most);
        let count = 0;
        for (; count < most && loaded.exports[String(count)]; count++) {
          const at = free.length ? free.pop() : table().grow(1);
          table().set(at, loaded.exports[String(count)]);
          out[count] = at;
        }
        return count;
      } catch (error) {
        if (globalThis.console) console.error(`entelechy.load: ${error.message}`);
        return -1;
      }
    },
    unload(slots, count) {
      const exports = instance().exports;
      for (const at of new Uint32Array(exports.memory.buffer, slots, count)) {
        table().set(at, null);
        free.push(at);
      }
    },
    now: () => performance.now(),
    fetch(path, length, from, out, bytes) {
      const exports = instance().exports;
      const name = new TextDecoder().decode(new Uint8Array(exports.memory.buffer, path, length));
      const base = globalThis.entelechy_files_base;
      if (!base) return -1;
      try {
        // Asked of the page's own server, which keeps them, as they are
        // read: anew from the start, and further on from what the last
        // read of the file found, kept until a read reaches its end.
        if (from === 0 || fetched?.name !== name) {
          fetched = null;
          const request = new XMLHttpRequest();
          request.open("GET", new URL(name, base).href, false);
          const worker = typeof WorkerGlobalScope !== "undefined";
          if (worker) request.responseType = "arraybuffer";
          else request.overrideMimeType("text/plain; charset=x-user-defined");
          request.send();
          if (request.status !== 200) return -1;
          let data;
          if (worker) data = new Uint8Array(request.response);
          else {
            const text = request.responseText;
            data = new Uint8Array(text.length);
            for (let i = 0; i < data.length; i++) data[i] = text.charCodeAt(i) & 0xff;
          }
          fetched = { name, data };
        }
        const { data } = fetched;
        const got = data.subarray(Math.min(from, data.length), Math.min(from + bytes, data.length));
        new Uint8Array(exports.memory.buffer, out, got.length).set(got);
        if (from + bytes >= data.length) fetched = null;
        return got.length;
      } catch (error) {
        if (globalThis.console) console.error(`entelechy.fetch: ${name}: ${error.message}`);
        return -1;
      }
    },
    fetch_start(path, length) {
      const exports = instance().exports;
      const name = new TextDecoder().decode(new Uint8Array(exports.memory.buffer, path, length));
      const ticket = ++tickets;
      const base = globalThis.entelechy_files_base;
      if (!base) {
        coming.set(ticket, -1);
        return ticket;
      }
      coming.set(ticket, -2);
      globalThis
        .fetch(new URL(name, base).href)
        .then((response) => (response.ok ? response.arrayBuffer() : Promise.reject(new Error(String(response.status)))))
        .then(
          (buffer) => coming.has(ticket) && coming.set(ticket, new Uint8Array(buffer)),
          () => coming.has(ticket) && coming.set(ticket, -1),
        );
      return ticket;
    },
    fetch_state(ticket) {
      const held = coming.get(ticket);
      return held === undefined ? -1 : typeof held === "number" ? held : held.length;
    },
    fetch_copy(ticket, from, out, bytes) {
      const held = coming.get(ticket);
      if (!(held instanceof Uint8Array)) return -1;
      const got = held.subarray(Math.min(from, held.length), Math.min(from + bytes, held.length));
      new Uint8Array(instance().exports.memory.buffer, out, got.length).set(got);
      return got.length;
    },
    fetch_end(ticket) {
      coming.delete(ticket);
    },
    wake: () => parked.wake(),
    wait:
      typeof WebAssembly.Suspending === "function"
        ? new WebAssembly.Suspending(() => parked.park())
        : () => {
            throw new Error("this browser cannot suspend WebAssembly (JavaScript Promise Integration)");
          },
  };
}

/// The one Entelechy a page loads, by the URL of the glue `wasm-bindgen`
/// wrote for it (`entelechy_ffi.js`, beside `entelechy_ffi_bg.wasm`). Its imports
/// from `entelechy` are `entelechy-host.mjs`, which an import map names.
const loaded = new Map();

/// Entelechy itself in the page: the session as WebAssembly (the C ABI,
/// `entelechy_ffi.wasm` built for `wasm32-unknown-unknown`), its documents the
/// files handed in, no disk and no server. It answers the same protocol
/// lines a socket carries, through `entelechy_request`, so a Driver runs it as
/// it runs a connection. A call that waits for the device suspends Entelechy
/// and answers when the device has; each session is its own, over the
/// one instance the page loaded.
class EntelechyInPage {
  constructor(exports) {
    this.x = exports;
    // What may wait for a device answers a promise.
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(exports[name]) : exports[name];
    this.open_files = promising("entelechy_open_files");
    this.open_replay = promising("entelechy_open_replay");
    this.request = promising("entelechy_request");
    this.presenting = promising("entelechy_present");
    this.turn = promising("entelechy_play_turn");
    this.files = exports.entelechy_files();
    this.session = 0;
    this.onclose = null;
    // One call at a time: a call waiting on the device has Entelechy
    // suspended, and another reaching into it then would find a session
    // half through its frame.
    this.line = Promise.resolve();
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
  }

  /// Loads Entelechy from its glue's URL, once a page, and makes a new set of
  /// files for a session of its own. `wasm` is its WebAssembly's bytes where
  /// the page carries them itself (a page exported whole), else it is
  /// fetched from beside the glue.
  static async load(glue, host, wasm) {
    const url = new URL(glue, location.href).href;
    if (!loaded.has(url)) {
      loaded.set(
        url,
        (async () => {
          // A page names the host module `entelechy` in its import map; a
          // worker has none, so the glue's imports are pointed at `host`
          // (the URL of entelechy-host.mjs) and its WebAssembly beside it.
          if (host) {
            const at = new URL(host, location.href).href;
            const text = (await (await fetch(url)).text()).replaceAll('from "entelechy"', `from ${JSON.stringify(at)}`);
            const glued = URL.createObjectURL(new Blob([text], { type: "text/javascript" }));
            const [module, hosted] = await Promise.all([import(glued), import(at)]);
            const wasmUrl = new URL(`entelechy_ffi_bg.wasm${new URL(url).search}`, url);
            const exports = await module.default({ module_or_path: wasm || wasmUrl });
            hosted.bind({ exports });
            return exports;
          }
          const [module, hosted] = await Promise.all([import(url), import("entelechy")]);
          const wasmUrl = new URL(`entelechy_ffi_bg.wasm${new URL(url).search}`, url);
          const exports = await module.default({ module_or_path: wasm || wasmUrl });
          hosted.bind({ exports });
          return exports;
        })(),
      );
    }
    return new EntelechyInPage(await loaded.get(url));
  }

  memory() {
    return new Uint8Array(this.x.memory.buffer);
  }

  /// `bytes` copied into Entelechy's memory for one call, which may answer a
  /// promise: the room is given back when it settles.
  withBytes(bytes, use) {
    const pointer = this.x.entelechy_alloc(bytes.length);
    if (!pointer) throw new Error("Entelechy has no room for the call");
    this.memory().set(bytes, pointer);
    const give = () => this.x.entelechy_free(pointer, bytes.length);
    let result;
    try {
      result = use(pointer, bytes.length);
    } catch (error) {
      give();
      throw error;
    }
    if (result instanceof Promise) return result.finally(give);
    give();
    return result;
  }

  withText(text, use) {
    return this.withBytes(this.encoder.encode(`${text}\0`), (pointer) => use(pointer));
  }

  text(pointer) {
    const memory = this.memory();
    let end = pointer;
    while (memory[end]) end++;
    return this.decoder.decode(memory.subarray(pointer, end));
  }

  /// Puts a file at `path` (`examples/text/scene.esn`), replacing what
  /// was there; an open session reads the change when asked to `sync`,
  /// as it reads a file changed on disk.
  put(path, bytes) {
    this.withText(path, (name) =>
      this.withBytes(bytes, (data, length) => this.x.entelechy_files_put(this.files, name, data, length)),
    );
  }

  /// Puts a file of `length` bytes the page keeps at `path` under
  /// `globalThis.entelechy_files_base`: read from there whenever it is
  /// read, never held here.
  putElsewhere(path, length) {
    this.withText(path, (name) => this.x.entelechy_files_put_elsewhere(this.files, name, length));
  }

  /// Opens `scene`, a path among the files put: on the JIT, whose code
  /// this page loads, or on `gpu`, the browser's WebGPU, or `cpu`.
  /// Resolves once it has loaded: in a page its Kernels prepare a few a
  /// sync, the page given a turn between syncs.
  async open(scene, backend = "jit") {
    if (this.session) this.x.entelechy_close(this.session);
    this.session = await this.withText(scene, (name) =>
      this.withText(backend, (on) => this.open_files(this.files, name, on)),
    );
    if (!this.session) throw new Error(this.text(this.x.entelechy_open_error()));
    await this.loaded();
  }

  /// Takes turns of loading until no Kernel waits to prepare, giving the
  /// page a turn between them; `onpreparing`, when set, hears how many wait.
  async loaded() {
    let left = this.x.entelechy_play_preparing(this.session);
    while (left > 0) {
      if (this.onpreparing) this.onpreparing(left);
      await new Promise((turn) => setTimeout(turn, 0));
      left = await this.turn(this.session);
      if (left < 0) throw new Error(this.text(this.x.entelechy_error(this.session)));
    }
    if (this.onpreparing) this.onpreparing(0);
  }

  /// Opens the scene the replay at `path` (among the files put) names,
  /// and plays it: see `EntelechyPlayer`.
  async openReplay(path, backend = "jit") {
    if (this.session) this.x.entelechy_close(this.session);
    this.session = await this.withText(path, (name) =>
      this.withText(backend, (on) => this.open_replay(this.files, name, on)),
    );
    if (!this.session) throw new Error(this.text(this.x.entelechy_open_error()));
    await this.loaded();
  }

  /// One request, answered in place; the same promise a connection's
  /// `call` gives. A panic inside the module ends it, as a closed
  /// connection ends a session.
  call(op, fields = {}) {
    return this.inTurn(() => this.callNow(op, fields));
  }

  /// `work` once every call before it has settled.
  inTurn(work) {
    const done = this.line.then(work);
    this.line = done.catch(() => {});
    return done;
  }

  async callNow(op, fields) {
    if (!this.session) throw new Error("no scene is open");
    let response;
    try {
      const line = JSON.stringify({ op, ...fields });
      const answer = await this.withText(line, (request) => this.request(this.session, request));
      response = JSON.parse(this.text(answer));
    } catch (error) {
      if (error instanceof WebAssembly.RuntimeError) {
        this.session = 0;
        if (this.onclose) this.onclose();
      }
      throw error;
    }
    if ("error" in response) {
      const error = new Error(response.error.message);
      error.line = response.error.line;
      error.column = response.error.column;
      throw error;
    }
    return response.ok;
  }

  /// Draws output `name` (the first, when none) on the canvas whose
  /// element id is `canvas`, from where the GPU holds it: whether it was
  /// drawn. Not drawn (a session not on the GPU, an output that is not a
  /// picture), it is to be read instead.
  present(canvas, name) {
    return this.inTurn(() => this.presentNow(canvas, name));
  }

  async presentNow(canvas, name) {
    if (!this.session) throw new Error("no scene is open");
    const done = await this.withText(canvas, (id) =>
      name === undefined
        ? this.presenting(this.session, id, 0)
        : this.withText(name, (output) => this.presenting(this.session, id, output)),
    );
    if (done < 0) throw new Error(this.text(this.x.entelechy_error(this.session)));
    return done === 0;
  }

  /// The pointer's look when the last frame changed it
  /// (`entelechy_play_pointer_look`): `null` when it did not, `"system"`
  /// for the system's own pointer, else the scene's picture as
  /// `{ rgba, width, height, hotspot }`, its bytes copied out.
  pointerLook() {
    // `EntelechyPointer`: a pointer and four counts, four bytes each.
    const look = this.x.entelechy_alloc(20);
    if (!look) return null;
    try {
      const said = this.x.entelechy_play_pointer_look(this.session, look);
      if (said === 2) return "system";
      if (said !== 0) return null;
      const [rgba, width, height, x, y] = new Uint32Array(this.x.memory.buffer, look, 5);
      const bytes = new Uint8ClampedArray(this.x.memory.buffer, rgba, width * height * 4).slice();
      return { rgba: bytes, width, height, hotspot: [x, y] };
    } finally {
      this.x.entelechy_free(look, 20);
    }
  }

  close() {
    if (this.session) this.x.entelechy_close(this.session);
    this.session = 0;
    if (this.onclose) this.onclose();
  }
}

/// Base64 to bytes.
function bytesOf(base64) {
  const text = atob(base64);
  const bytes = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
  return bytes;
}

/// A `read` in `ppm` format as RGBA pixels the canvas can take.
function decodePpm(base64) {
  const bytes = bytesOf(base64);
  // "P6\n<width> <height>\n255\n" then the pixels.
  let cursor = 0;
  const fields = [];
  while (fields.length < 4) {
    while (cursor < bytes.length && /\s/.test(String.fromCharCode(bytes[cursor]))) cursor++;
    let start = cursor;
    while (cursor < bytes.length && !/\s/.test(String.fromCharCode(bytes[cursor]))) cursor++;
    fields.push(String.fromCharCode(...bytes.subarray(start, cursor)));
  }
  cursor++;
  if (fields[0] !== "P6") throw new Error(`not a binary PPM: ${fields[0]}`);
  const width = Number(fields[1]);
  const height = Number(fields[2]);
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0, j = cursor; i < width * height; i++, j += 3) {
    data[i * 4] = bytes[j];
    data[i * 4 + 1] = bytes[j + 1];
    data[i * 4 + 2] = bytes[j + 2];
    data[i * 4 + 3] = 255;
  }
  return { width, height, data };
}

/// A `read` of a Field that is not pixels: its words as little-endian
/// 32-bit values, readable as floats or ints.
function decodeWords(base64) {
  const bytes = bytesOf(base64);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const count = bytes.byteLength / 4;
  return {
    count,
    float: (i) => view.getFloat32(i * 4, true),
    int: (i) => view.getInt32(i * 4, true),
    bool: (i) => view.getUint32(i * 4, true) !== 0,
  };
}

/// Where a frame of `width` by `height` lands in a `targetWidth` by
/// `targetHeight` canvas, aspect kept, centered: the Player's placement.
function placement(width, height, targetWidth, targetHeight) {
  if (!width || !height || !targetWidth || !targetHeight) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  const scale = Math.min(targetWidth / width, targetHeight / height);
  const w = Math.max(1, Math.floor(width * scale));
  const h = Math.max(1, Math.floor(height * scale));
  return { x: Math.floor((targetWidth - w) / 2), y: Math.floor((targetHeight - h) / 2), width: w, height: h };
}

/// Drives a session from a canvas: the canvas is the window (its size in
/// device pixels and the device's scale go with every `evaluate`), the
/// pointer, buttons, wheel, keys and the text typed over it are the input,
/// and each frame
/// is an `evaluate` followed by a `read` drawn onto the canvas. Given
/// `shown`, a second canvas over the first that takes no pointer, a
/// session that can present (Entelechy in the page, on its GPU) draws each
/// frame there from the device instead, and nothing is read back.
class Driver {
  constructor(client, canvas, shown = null) {
    this.client = client;
    this.canvas = canvas;
    this.shown = shown;
    this.presented = false;
    this.context = canvas.getContext("2d");
    this.pointer = [0, 0];
    this.buttons = new Set();
    this.keys = new Set();
    this.typing = [];
    this.wheel = [0, 0];
    this.frame = { width: 0, height: 0 };
    this.placed = { x: 0, y: 0, width: 0, height: 0 };
    this.running = false;
    this.started = performance.now();
    this.lastMillis = 0;
    this.onframe = null;
    this.onerror = null;
    this.listen();
  }

  /// The key codes a Kernel reads (`entelechy help key`): the key pressed
  /// as a US keyboard has it, whatever its modifiers type (Option+Z is Z,
  /// not Ω): a letter or a digit by its ASCII uppercase, a punctuation key
  /// by its code; else a character typed with shift by the key under it,
  /// a named key by the browser's code.
  static keyCode(event) {
    const code = event.code || "";
    if (/^Key[A-Z]$/.test(code)) return code.charCodeAt(3);
    if (/^Digit[0-9]$/.test(code)) return code.charCodeAt(5);
    if (code in Driver.CODES) return Driver.CODES[code];
    if (event.key.length === 1) {
      const upper = event.key.toUpperCase();
      if (/^[A-Z0-9 ]$/.test(upper)) return upper.charCodeAt(0);
      return Driver.TYPED[event.key];
    }
    return Driver.NAMED[event.key];
  }

  static CODES = {
    Space: 32, Semicolon: 186, Equal: 187, Comma: 188, Minus: 189, Period: 190, Slash: 191,
    Backquote: 192, BracketLeft: 219, Backslash: 220, BracketRight: 221, Quote: 222,
  };

  static TYPED = {
    ")": 48, "!": 49, "@": 50, "#": 51, "$": 52, "%": 53, "^": 54, "&": 55, "*": 56, "(": 57,
    ";": 186, ":": 186, "=": 187, "+": 187, ",": 188, "<": 188, "-": 189, "_": 189,
    ".": 190, ">": 190, "/": 191, "?": 191, "`": 192, "~": 192, "[": 219, "{": 219,
    "\\": 220, "|": 220, "]": 221, "}": 221, "'": 222, "\"": 222,
  };

  static NAMED = {
    Backspace: 8, Tab: 9, Enter: 13, Shift: 16, Control: 17, Alt: 18, Escape: 27,
    PageUp: 33, PageDown: 34, End: 35, Home: 36,
    ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Delete: 46, Meta: 91,
  };

  listen() {
    const canvas = this.canvas;
    canvas.tabIndex = 0;
    canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    canvas.addEventListener("pointermove", (event) => this.moved(event));
    canvas.addEventListener("pointerdown", (event) => {
      canvas.focus();
      canvas.setPointerCapture(event.pointerId);
      this.buttons.add(Driver.buttonNumber(event.button));
      this.moved(event);
    });
    canvas.addEventListener("pointerup", (event) => {
      this.buttons.delete(Driver.buttonNumber(event.button));
      this.moved(event);
    });
    canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      // Lines of scrolling, y positive away from the person, as the Player counts.
      const unit = event.deltaMode === 1 ? 1 : 1 / 40;
      this.wheel[0] += event.deltaX * unit;
      this.wheel[1] -= event.deltaY * unit;
    }, { passive: false });
    canvas.addEventListener("keydown", (event) => {
      // A character typed, unless it is a shortcut or an input method's.
      if (event.key.length >= 1 && [...event.key].length === 1 && !event.isComposing
        && !event.ctrlKey && !event.metaKey && !/\p{Cc}/u.test(event.key)) {
        this.typing.push(event.key);
      }
      const code = Driver.keyCode(event);
      if (code !== undefined) {
        this.keys.add(code);
        event.preventDefault();
      }
    });
    // What an input method commits is typed.
    canvas.addEventListener("compositionend", (event) => {
      this.typing.push(...[...(event.data || "")]);
    });
    canvas.addEventListener("keyup", (event) => {
      const code = Driver.keyCode(event);
      if (code !== undefined) this.keys.delete(code);
    });
    canvas.addEventListener("blur", () => {
      this.keys.clear();
      this.buttons.clear();
    });
  }

  /// A browser button to a protocol one: 0 left, 1 right, 2 middle.
  static buttonNumber(button) {
    return button === 1 ? 2 : button === 2 ? 1 : 0;
  }

  /// The pointer in output pixels: the canvas position mapped through
  /// where the frame is placed, clamped to the frame like the Player.
  moved(event) {
    const rect = this.canvas.getBoundingClientRect();
    const scale = this.canvas.width / rect.width;
    const x = (event.clientX - rect.left) * scale;
    const y = (event.clientY - rect.top) * scale;
    const p = this.placed;
    if (!p.width || !p.height || !this.frame.width) {
      this.pointer = [0, 0];
      return;
    }
    const fx = ((x - p.x) * this.frame.width) / p.width;
    const fy = ((y - p.y) * this.frame.height) / p.height;
    this.pointer = [
      Math.min(Math.max(fx, 0), this.frame.width - 1),
      Math.min(Math.max(fy, 0), this.frame.height - 1),
    ];
  }

  /// The window the session sees: the canvas in device pixels.
  window() {
    return [this.canvas.width, this.canvas.height];
  }

  /// One frame: evaluate under the current input, read the picture, draw it.
  async step() {
    const wheel = this.wheel;
    this.wheel = [0, 0];
    // As many characters as an evaluation holds; the rest the next.
    const text = this.typing.splice(0, 32).join("");
    const started = performance.now();
    const report = await this.client.call("evaluate", {
      time: (started - this.started) / 1000,
      pointer: this.pointer,
      buttons: [...this.buttons],
      wheel,
      keys: [...this.keys],
      ...(text ? { text } : {}),
      window: this.window(),
      scale: window.devicePixelRatio || 1,
    });
    const picture = report.outputs.find((output) => output.width > 0);
    this.presented = picture ? await this.present(picture) : false;
    if (picture && !this.presented) {
      const read = await this.client.call("read", { output: picture.name, format: "ppm" });
      const decoded = decodePpm(read.data);
      this.draw(decoded);
    }
    this.lastMillis = performance.now() - started;
    if (this.onframe) this.onframe(report, this.lastMillis);
    return report;
  }

  /// Draws `picture` on the second canvas from the device, where the
  /// session can: whether it did.
  async present(picture) {
    const shown = this.shown;
    if (!shown || typeof this.client.present !== "function") return false;
    shown.width = this.canvas.width;
    shown.height = this.canvas.height;
    const drawn = await this.client.present(shown.id, picture.name);
    shown.style.display = drawn ? "block" : "none";
    if (drawn) {
      this.frame = { width: picture.width, height: picture.height };
      this.placed = placement(picture.width, picture.height, shown.width, shown.height);
    }
    return drawn;
  }

  draw(decoded) {
    const { width, height, data } = decoded;
    if (this.frame.width !== width || this.frame.height !== height) {
      this.frame = { width, height };
      this.scratch = document.createElement("canvas");
      this.scratch.width = width;
      this.scratch.height = height;
    }
    const image = new ImageData(data, width, height);
    this.scratch.getContext("2d").putImageData(image, 0, 0);
    const canvas = this.canvas;
    const placed = placement(width, height, canvas.width, canvas.height);
    this.placed = placed;
    const context = this.context;
    context.imageSmoothingEnabled = false;
    context.fillStyle = "#000";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.drawImage(this.scratch, placed.x, placed.y, placed.width, placed.height);
  }

  /// Frame after frame until `stop`; an error stops the loop and is reported.
  async run() {
    if (this.running) return;
    this.running = true;
    while (this.running) {
      try {
        await this.step();
      } catch (error) {
        this.running = false;
        if (this.onerror) this.onerror(error);
        return;
      }
      await new Promise((resolve) => requestAnimationFrame(resolve));
    }
  }

  stop() {
    this.running = false;
  }
}

globalThis.Entelechy = { EntelechyClient, EntelechyInPage, host, makeWebGlHost, Driver, decodePpm, decodeWords, placement, bytesOf };
globalThis.entelechy_webgl ??= makeWebGlHost();

function makeWebGlHost() {
  const VERTEX_SHADER = `#version 300 es
precision highp float;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
`;
  const FAIL_REDUCE_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform highp usampler2D u_fail;
uniform highp uvec2 u_extent;
layout(location = 0) out highp uvec4 out_reduced;
void main() {
    uvec2 gid = uvec2(gl_FragCoord.xy);
    uint best = 0u;
    uint lo = 0u;
    uint hi = 0u;
    for (uint y = gid.y; y < u_extent.y; y += 16u) {
        for (uint x = gid.x; x < u_extent.x; x += 16u) {
            uvec4 v = texelFetch(u_fail, ivec2(int(x), int(y)), 0);
            if (v.x > best) { best = v.x; }
            uint next_lo = lo + v.y;
            hi += v.z + ((next_lo < lo) ? 1u : 0u);
            lo = next_lo;
        }
    }
    out_reduced = uvec4(best, lo, hi, 0u);
}
`;
  const SAME_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform highp usampler2D u_a;
uniform highp usampler2D u_b;
uniform highp uint u_words;
uniform highp uvec2 u_extent;
layout(location = 0) out highp uvec4 out_diff;
void main() {
    uvec2 gid = uvec2(gl_FragCoord.xy);
    uint diff = 0u;
    for (uint y = gid.y; y < u_extent.y; y += 16u) {
        for (uint x = gid.x; x < u_extent.x; x += 16u) {
            uint base = ((y << 11u) | x) << 2u;
            if (base >= u_words) { continue; }
            uvec4 a = texelFetch(u_a, ivec2(int(x), int(y)), 0);
            uvec4 b = texelFetch(u_b, ivec2(int(x), int(y)), 0);
            uint rem = u_words - base;
            if (a.x != b.x) { diff = 1u; }
            if (rem > 1u && a.y != b.y) { diff = 1u; }
            if (rem > 2u && a.z != b.z) { diff = 1u; }
            if (rem > 3u && a.w != b.w) { diff = 1u; }
        }
    }
    out_diff = uvec4(diff, 0u, 0u, 0u);
}
`;
  const BLIT_SHADER = `#version 300 es
precision highp float;
precision highp int;
precision highp usampler2D;
uniform highp usampler2D u_image;
uniform highp usampler2D u_overlay;
uniform highp vec4 u_frame[3];
layout(location = 0) out vec4 out_color;
void main() {
    vec2 pos = vec2(gl_FragCoord.x, u_frame[0].y - gl_FragCoord.y);
    if (pos.x < u_frame[2].y && pos.y < u_frame[2].z) {
        uint idx = uint(pos.y) * uint(u_frame[2].y) + uint(pos.x);
        uint t = idx >> 2u;
        uvec4 ov = texelFetch(u_overlay, ivec2(int(t & 2047u), int(t >> 11u)), 0);
        uint lane = idx & 3u;
        uint o = (lane == 0u) ? ov.x : ((lane == 1u) ? ov.y : ((lane == 2u) ? ov.z : ov.w));
        vec3 c = vec3(float((o >> 16u) & 255u), float((o >> 8u) & 255u), float(o & 255u)) / 255.0;
        out_color = vec4(c, 1.0);
        return;
    }
    vec2 p = pos - u_frame[1].xy;
    if (p.x < 0.0 || p.y < 0.0 || p.x >= u_frame[1].z || p.y >= u_frame[1].w) {
        out_color = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    uint w = uint(u_frame[0].z);
    uint h = uint(u_frame[0].w);
    uint x = min(uint(p.x * u_frame[0].z / u_frame[1].z), w - 1u);
    uint y = min(uint(p.y * u_frame[0].w / u_frame[1].w), h - 1u);
    uint pixel = y * w + x;
    uvec4 raw_bits = texelFetch(u_image, ivec2(int(pixel & 2047u), int(pixel >> 11u)), 0);
    uvec3 rgb_bits = raw_bits.xyz;
    vec3 raw = uintBitsToFloat(rgb_bits);
    bvec3 nan = greaterThan(rgb_bits & uvec3(0x7FFFFFFFu), uvec3(0x7F800000u));
    vec3 clean = mix(raw, vec3(0.0), nan);
    out_color = vec4(clamp(clean, vec3(0.0), vec3(1.0)), 1.0);
}
`;

  let gl = null;
  let info = null;
  let defaultFbo = null;
  let vertShader = null;
  let emptyVao = null;
  let drawFbo = null;
  let readFbo = null;
  let ctxUbo = null;
  let paramsUbo = null;
  let ctxCap = 0;
  let paramsCap = 0;
  let zeroTex = null;
  let reduceTex = null;
  let reduceProg = null;
  let reduceFailExtentLoc = null;
  let sameProg = null;
  let sameWordsLoc = null;
  let sameExtentLoc = null;
  let blitProg = null;
  let blitFrameLoc = null;
  let overlayTex = null;
  let overlayCapWords = 0;

  let nextTexId = 1;
  let nextProgId = 1;
  const textures = new Map();
  const programs = new Map();
  const tmp4 = new Uint32Array(4);
  const reduceBuf = new Uint32Array(16 * 16 * 4);

  function findCanvas(name) {
    if (globalThis.entelechy_canvases && globalThis.entelechy_canvases[name]) {
      return globalThis.entelechy_canvases[name];
    }
    if (typeof document !== "undefined" && name) {
      return document.getElementById(name);
    }
    return null;
  }

  function makeRawTexture(w, h) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, w, h);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return tex;
  }

  function buildShader(type, source) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh) || "shader compile failed";
      gl.deleteShader(sh);
      return { error: log.trim() };
    }
    return { shader: sh };
  }

  function linkUtility(fragSource) {
    const built = buildShader(gl.FRAGMENT_SHADER, fragSource);
    if (built.error) throw new Error(built.error);
    const prog = gl.createProgram();
    gl.attachShader(prog, vertShader);
    gl.attachShader(prog, built.shader);
    gl.linkProgram(prog);
    gl.deleteShader(built.shader);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog) || "program link failed";
      gl.deleteProgram(prog);
      throw new Error(log.trim());
    }
    return prog;
  }

  function readSpanWords(entry, startWord, wordCount) {
    const firstTexel = startWord >>> 2;
    const lastTexel = (startWord + wordCount - 1) >>> 2;
    const w = entry.width;
    const y0 = (firstTexel / w) | 0;
    const y1 = (lastTexel / w) | 0;
    gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, entry.tex, 0);
    let buf, off;
    if (y0 === y1) {
      const x0 = firstTexel % w;
      const cols = lastTexel - firstTexel + 1;
      buf = new Uint32Array(cols * 4);
      gl.readPixels(x0, y0, cols, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, buf);
      off = startWord & 3;
    } else {
      const rows = y1 - y0 + 1;
      buf = new Uint32Array(w * rows * 4);
      gl.readPixels(0, y0, w, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, buf);
      off = startWord - y0 * w * 4;
    }
    gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
    return buf.subarray(off, off + wordCount);
  }

  function uploadInto(tex, w, startWord, words) {
    if (!words.length) return;
    let cur = startWord >>> 0;
    let srcAt = 0;
    let left = words.length >>> 0;

    const patchTexel = (texelIdx, laneStart, count) => {
      const tx = texelIdx % w;
      const ty = (texelIdx / w) | 0;
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
      gl.readPixels(tx, ty, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tmp4);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
      for (let k = 0; k < count; k++) tmp4[laneStart + k] = words[srcAt + k];
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, tx, ty, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tmp4);
    };

    if ((cur & 3) !== 0) {
      const lane = cur & 3;
      const take = Math.min(4 - lane, left);
      patchTexel(cur >>> 2, lane, take);
      cur += take;
      srcAt += take;
      left -= take;
    }

    const fullTexels = left >>> 2;
    if (fullTexels > 0) {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      let tCur = cur >>> 2;
      let tLeft = fullTexels;
      const tx0 = tCur % w;
      if (tx0 !== 0) {
        const cols = Math.min(w - tx0, tLeft);
        const ty0 = (tCur / w) | 0;
        gl.texSubImage2D(gl.TEXTURE_2D, 0, tx0, ty0, cols, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words, srcAt);
        tCur += cols;
        srcAt += cols * 4;
        tLeft -= cols;
      }
      const rows = (tLeft / w) | 0;
      if (rows > 0) {
        const ty = (tCur / w) | 0;
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, ty, w, rows, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words, srcAt);
        const span = rows * w;
        tCur += span;
        srcAt += span * 4;
        tLeft -= span;
      }
      if (tLeft > 0) {
        const ty = (tCur / w) | 0;
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, ty, tLeft, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, words, srcAt);
        tCur += tLeft;
        srcAt += tLeft * 4;
      }
      cur += fullTexels * 4;
      left -= fullTexels * 4;
    }

    if (left > 0) {
      const tLast = cur >>> 2;
      if (startWord === 0 && cur + left >= words.length) {
        tmp4[0] = left > 0 ? words[srcAt] : 0;
        tmp4[1] = left > 1 ? words[srcAt + 1] : 0;
        tmp4[2] = left > 2 ? words[srcAt + 2] : 0;
        tmp4[3] = 0;
        const tx = tLast % w;
        const ty = (tLast / w) | 0;
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, tx, ty, 1, 1, gl.RGBA_INTEGER, gl.UNSIGNED_INT, tmp4);
      } else {
        patchTexel(tLast, 0, left);
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  return {
    open() {
      const desired = globalThis.entelechy_webgl_canvas || findCanvas("shown");
      if (gl && !gl.isContextLost() && (!desired || gl.canvas === desired)) {
        return info;
      }
      try {
        const opts = {
          alpha: false,
          depth: false,
          stencil: false,
          antialias: false,
          premultipliedAlpha: false,
          preserveDrawingBuffer: false,
          powerPreference: "high-performance",
        };
        const makeFallback = () =>
          typeof OffscreenCanvas !== "undefined"
            ? new OffscreenCanvas(1, 1)
            : typeof document !== "undefined"
              ? document.createElement("canvas")
              : null;
        let ctx = desired && typeof desired.getContext === "function" ? desired.getContext("webgl2", opts) : null;
        if (!ctx) {
          const fb = makeFallback();
          if (!fb || typeof fb.getContext !== "function") {
            return { error: "no canvas available for WebGL2" };
          }
          ctx = fb.getContext("webgl2", opts);
        }
        if (!ctx) return { error: "WebGL2 context creation refused" };

        gl = ctx;
        defaultFbo = gl.getParameter(gl.FRAMEBUFFER_BINDING) || null;
        textures.clear();
        programs.clear();
        overlayTex = null;
        overlayCapWords = 0;
        ctxCap = 0;
        paramsCap = 0;

        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
        gl.pixelStorei(gl.PACK_ALIGNMENT, 4);
        gl.disable(gl.DEPTH_TEST);
        gl.disable(gl.BLEND);
        gl.disable(gl.CULL_FACE);
        gl.disable(gl.SCISSOR_TEST);

        const vs = buildShader(gl.VERTEX_SHADER, VERTEX_SHADER);
        if (vs.error) return { error: vs.error };
        vertShader = vs.shader;
        emptyVao = gl.createVertexArray();
        drawFbo = gl.createFramebuffer();
        readFbo = gl.createFramebuffer();
        ctxUbo = gl.createBuffer();
        paramsUbo = gl.createBuffer();
        zeroTex = makeRawTexture(1, 1);
        reduceTex = makeRawTexture(16, 16);

        reduceProg = linkUtility(FAIL_REDUCE_SHADER);
        gl.useProgram(reduceProg);
        gl.uniform1i(gl.getUniformLocation(reduceProg, "u_fail"), 0);
        reduceFailExtentLoc = gl.getUniformLocation(reduceProg, "u_extent");

        sameProg = linkUtility(SAME_SHADER);
        gl.useProgram(sameProg);
        gl.uniform1i(gl.getUniformLocation(sameProg, "u_a"), 0);
        gl.uniform1i(gl.getUniformLocation(sameProg, "u_b"), 1);
        sameWordsLoc = gl.getUniformLocation(sameProg, "u_words");
        sameExtentLoc = gl.getUniformLocation(sameProg, "u_extent");

        blitProg = linkUtility(BLIT_SHADER);
        gl.useProgram(blitProg);
        gl.uniform1i(gl.getUniformLocation(blitProg, "u_image"), 0);
        gl.uniform1i(gl.getUniformLocation(blitProg, "u_overlay"), 1);
        blitFrameLoc = gl.getUniformLocation(blitProg, "u_frame");
        gl.useProgram(null);

        const dbg = gl.getExtension("WEBGL_debug_renderer_info");
        const renderer =
          (dbg && gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) ||
          gl.getParameter(gl.RENDERER) ||
          "WebGL2";
        info = {
          renderer: `${renderer} (WebGL2)`,
          maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) || 2048,
          maxTextureUnits: gl.getParameter(gl.MAX_TEXTURE_IMAGE_UNITS) || 16,
          maxDrawBuffers: gl.getParameter(gl.MAX_DRAW_BUFFERS) || 4,
          maxUniformBlockBytes: gl.getParameter(gl.MAX_UNIFORM_BLOCK_SIZE) || 16384,
        };
        return info;
      } catch (err) {
        gl = null;
        return { error: err && err.message ? err.message : String(err) };
      }
    },

    createTexture(width, height) {
      const tex = makeRawTexture(width, height);
      const id = nextTexId++;
      textures.set(id, { tex, width, height });
      return id;
    },

    deleteTexture(id) {
      const entry = textures.get(id);
      if (!entry) return;
      textures.delete(id);
      if (gl) gl.deleteTexture(entry.tex);
    },

    uploadWords(id, startWord, words) {
      const entry = textures.get(id);
      if (!entry) return;
      uploadInto(entry.tex, entry.width, startWord, words);
    },

    copyWords(srcId, srcStartWord, dstId, dstStartWord, wordCount) {
      if (!wordCount) return;
      const src = textures.get(srcId);
      const dst = textures.get(dstId);
      if (!src || !dst) return;
      if (
        srcStartWord === 0 &&
        dstStartWord === 0 &&
        src.width === dst.width &&
        src.height === dst.height &&
        wordCount >= (src.width * src.height - 1) * 4
      ) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
        gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, src.tex, 0);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, drawFbo);
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, dst.tex, 0);
        gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
        gl.blitFramebuffer(0, 0, src.width, src.height, 0, 0, dst.width, dst.height, gl.COLOR_BUFFER_BIT, gl.NEAREST);
        gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
        gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
        return;
      }
      const span = readSpanWords(src, srcStartWord >>> 0, wordCount >>> 0);
      uploadInto(dst.tex, dst.width, dstStartWord >>> 0, span);
    },

    readWords(id, wordLen) {
      const entry = textures.get(id);
      if (!entry || !wordLen) return new Uint32Array(0);
      return readSpanWords(entry, 0, wordLen >>> 0);
    },

    compileProgram(fragSource, totalInputs, packedIndices, hasPacked, outputCount, hasFail) {
      const fs = buildShader(gl.FRAGMENT_SHADER, fragSource);
      if (fs.error) return fs.error;
      const prog = gl.createProgram();
      gl.attachShader(prog, vertShader);
      gl.attachShader(prog, fs.shader);
      gl.linkProgram(prog);
      gl.deleteShader(fs.shader);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(prog) || "program link failed";
        gl.deleteProgram(prog);
        return log.trim();
      }
      const ctxIdx = gl.getUniformBlockIndex(prog, "CtxBlock");
      if (ctxIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, ctxIdx, 0);
      const parIdx = gl.getUniformBlockIndex(prog, "ParamsBlock");
      if (parIdx !== gl.INVALID_INDEX) gl.uniformBlockBinding(prog, parIdx, 1);

      gl.useProgram(prog);
      const packedSet = new Set(packedIndices);
      const bindings = [];
      let unit = 0;
      for (let i = 0; i < totalInputs; i++) {
        if (packedSet.has(i)) continue;
        const loc = gl.getUniformLocation(prog, `u_in${i}`);
        if (loc !== null) {
          gl.uniform1i(loc, unit);
          bindings.push({ unit, inputIndex: i, packed: false });
          unit++;
        }
      }
      if (hasPacked) {
        const loc = gl.getUniformLocation(prog, "u_packed");
        if (loc !== null) {
          gl.uniform1i(loc, unit);
          bindings.push({ unit, inputIndex: -1, packed: true });
          unit++;
        }
      }
      gl.useProgram(null);

      const id = nextProgId++;
      const totalAtt = outputCount + (hasFail ? 1 : 0);
      const drawBuffers = [];
      for (let a = 0; a < totalAtt; a++) drawBuffers.push(gl.COLOR_ATTACHMENT0 + a);
      programs.set(id, { prog, bindings, drawBuffers });
      return id;
    },

    runPass(progId, width, height, ctxBytes, paramsBytes, inputIds, packedId, outputIds, failId) {
      const entry = programs.get(progId);
      if (!entry) return "unknown WebGL2 program";

      gl.bindBuffer(gl.UNIFORM_BUFFER, ctxUbo);
      if (ctxBytes.byteLength > ctxCap) {
        ctxCap = Math.max(ctxBytes.byteLength, 256);
        gl.bufferData(gl.UNIFORM_BUFFER, ctxCap, gl.DYNAMIC_DRAW);
      }
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, ctxBytes);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 0, ctxUbo);

      gl.bindBuffer(gl.UNIFORM_BUFFER, paramsUbo);
      if (paramsBytes.byteLength > paramsCap) {
        paramsCap = Math.max(paramsBytes.byteLength, 256);
        gl.bufferData(gl.UNIFORM_BUFFER, paramsCap, gl.DYNAMIC_DRAW);
      }
      gl.bufferSubData(gl.UNIFORM_BUFFER, 0, paramsBytes);
      gl.bindBufferBase(gl.UNIFORM_BUFFER, 1, paramsUbo);
      gl.bindBuffer(gl.UNIFORM_BUFFER, null);

      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, drawFbo);
      let att = 0;
      for (let i = 0; i < outputIds.length; i++) {
        const outTex = textures.get(outputIds[i]);
        if (!outTex) {
          for (let a = 0; a < att; a++) {
            gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + a, gl.TEXTURE_2D, null, 0);
          }
          gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
          return "missing output texture";
        }
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + att, gl.TEXTURE_2D, outTex.tex, 0);
        att++;
      }
      if (failId) {
        const fTex = textures.get(failId);
        if (!fTex) {
          for (let a = 0; a < att; a++) {
            gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + a, gl.TEXTURE_2D, null, 0);
          }
          gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
          return "missing failure texture";
        }
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + att, gl.TEXTURE_2D, fTex.tex, 0);
        att++;
      }
      gl.drawBuffers(entry.drawBuffers);

      for (const b of entry.bindings) {
        const tEntry = textures.get(b.packed ? packedId : inputIds[b.inputIndex]);
        gl.activeTexture(gl.TEXTURE0 + b.unit);
        gl.bindTexture(gl.TEXTURE_2D, tEntry ? tEntry.tex : zeroTex);
      }

      gl.viewport(0, 0, width, height);
      gl.useProgram(entry.prog);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      for (const b of entry.bindings) {
        gl.activeTexture(gl.TEXTURE0 + b.unit);
        gl.bindTexture(gl.TEXTURE_2D, null);
      }
      for (let a = 0; a < att; a++) {
        gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0 + a, gl.TEXTURE_2D, null, 0);
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
    },

    reduceFail(failId, width, height) {
      const fTex = textures.get(failId);
      if (!fTex) return new Uint32Array(4);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, drawFbo);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reduceTex, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, fTex.tex);
      gl.viewport(0, 0, 16, 16);
      gl.useProgram(reduceProg);
      gl.uniform2ui(reduceFailExtentLoc, width, height);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reduceTex, 0);
      gl.readPixels(0, 0, 16, 16, gl.RGBA_INTEGER, gl.UNSIGNED_INT, reduceBuf);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);

      let best = 0;
      let lo = 0;
      let hi = 0;
      for (let i = 0; i < 1024; i += 4) {
        const w = reduceBuf[i];
        if (w > best) best = w;
        const sumLo = (lo + reduceBuf[i + 1]) >>> 0;
        hi = (hi + reduceBuf[i + 2] + (sumLo < lo ? 1 : 0)) >>> 0;
        lo = sumLo;
      }
      return new Uint32Array([best, lo, hi, 0]);
    },

    sameTextures(idA, idB, wordLen) {
      const a = textures.get(idA);
      const b = textures.get(idB);
      if (!a || !b) return false;
      const texels = Math.max(1, Math.ceil(wordLen / 4));
      const tw = texels <= 2048 ? texels : 2048;
      const th = texels <= 2048 ? 1 : Math.ceil(texels / 2048);

      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, drawFbo);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reduceTex, 0);
      gl.drawBuffers([gl.COLOR_ATTACHMENT0]);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, a.tex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, b.tex);
      gl.viewport(0, 0, 16, 16);
      gl.useProgram(sameProg);
      gl.uniform1ui(sameWordsLoc, wordLen >>> 0);
      gl.uniform2ui(sameExtentLoc, tw, th);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.framebufferTexture2D(gl.DRAW_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);

      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, readFbo);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, reduceTex, 0);
      gl.readPixels(0, 0, 16, 16, gl.RGBA_INTEGER, gl.UNSIGNED_INT, reduceBuf);
      gl.framebufferTexture2D(gl.READ_FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, null, 0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);

      for (let i = 0; i < 1024; i += 4) {
        if (reduceBuf[i] !== 0) return false;
      }
      return true;
    },

    canvasSize(name) {
      const c = findCanvas(name) || (gl && gl.canvas);
      if (!c) return new Uint32Array(0);
      return new Uint32Array([c.width >>> 0, c.height >>> 0]);
    },

    present(name, imageId, windowWidth, windowHeight, frameFloats12, overlayWords) {
      const img = textures.get(imageId);
      if (!img || !gl) return false;
      const target = findCanvas(name) || gl.canvas;
      if (!target) return false;
      if (gl.canvas.width !== windowWidth || gl.canvas.height !== windowHeight) {
        gl.canvas.width = windowWidth;
        gl.canvas.height = windowHeight;
      }
      let ovTex = zeroTex;
      if (overlayWords && overlayWords.length > 0) {
        if (!overlayTex || overlayCapWords < overlayWords.length) {
          if (overlayTex) gl.deleteTexture(overlayTex.tex);
          const texels = Math.ceil(overlayWords.length / 4);
          const ow = texels <= 2048 ? texels : 2048;
          const oh = texels <= 2048 ? 1 : Math.ceil(texels / 2048);
          overlayTex = { tex: makeRawTexture(ow, oh), width: ow, height: oh };
          overlayCapWords = ow * oh * 4;
        }
        uploadInto(overlayTex.tex, overlayTex.width, 0, overlayWords);
        ovTex = overlayTex.tex;
      }

      gl.bindFramebuffer(gl.FRAMEBUFFER, defaultFbo);
      gl.viewport(0, 0, windowWidth, windowHeight);
      gl.useProgram(blitProg);
      gl.uniform4fv(blitFrameLoc, frameFloats12);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, img.tex);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, ovTex);
      gl.bindVertexArray(emptyVao);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, null);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, null);

      if (target !== gl.canvas) {
        const ctx2d = target.getContext && target.getContext("2d");
        if (!ctx2d) return false;
        if (target.width !== windowWidth || target.height !== windowHeight) {
          target.width = windowWidth;
          target.height = windowHeight;
        }
        ctx2d.drawImage(gl.canvas, 0, 0);
      }
      return true;
    },

    finish() {
      if (gl) gl.finish();
    },
  };
}
})();
