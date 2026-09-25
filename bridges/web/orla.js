// Orla's creation protocol from a browser. A client of the public
// contracts and nothing more: JSON lines as WebSocket text frames to a
// session serving with `--listen`, the same lines Blender or a shell
// would send on the socket; or the same lines answered in the page by
// Orla built as WebAssembly, through its C ABI.
//
// A classic script, not a module, so a page opened from a file works:
// browsers refuse module imports between file URLs. It defines `Orla`.

(() => {
/// A connection. The session answers one line per request, in order, so
/// answers are matched to requests first in, first out; the `id` a
/// request carries is echoed but not relied on, since `kernel` and
/// `retire` use that field for a Kernel's id.
class OrlaClient {
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
      socket.onopen = () => resolve(new OrlaClient(socket));
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

/// Parking for Orla while it waits for the device: `park` resolves once
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
        const resume = () => {
          if (waiting !== resume) return;
          waiting = null;
          woken = false;
          done();
        };
        waiting = resume;
        setTimeout(resume, 1000);
      });
    },
    wake() {
      woken = true;
      if (waiting) waiting();
    },
  };
}

/// What Orla built as WebAssembly imports from its host: `orla.load`,
/// which compiles a Kernel's code (WebAssembly the JIT wrote) over Orla's
/// memory and function table and puts its functions into the table;
/// `orla.unload`, which gives their slots back; `orla.wait`, which
/// suspends Orla (JavaScript Promise Integration) until `orla.wake`,
/// which the device's answer calls, which is how Orla waits for a
/// device; and `orla.now`, the page's clock in milliseconds.
/// `instance()` answers the Orla instance, once there is one.
function host(instance) {
  const free = [];
  const parked = parking();
  const table = () => instance().exports.__indirect_function_table;
  return {
    load(bytes, length, slots, most) {
      const exports = instance().exports;
      try {
        const code = new Uint8Array(exports.memory.buffer, bytes, length).slice();
        const loaded = new WebAssembly.Instance(new WebAssembly.Module(code), {
          env: { memory: exports.memory, table: table() },
        });
        const out = new Uint32Array(exports.memory.buffer, slots, most);
        let count = 0;
        for (; count < most && loaded.exports[String(count)]; count++) {
          const at = free.length ? free.pop() : table().grow(1);
          table().set(at, loaded.exports[String(count)]);
          out[count] = at;
        }
        return count;
      } catch (error) {
        if (globalThis.console) console.error(`orla.load: ${error.message}`);
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
    wake: () => parked.wake(),
    wait:
      typeof WebAssembly.Suspending === "function"
        ? new WebAssembly.Suspending(() => parked.park())
        : () => {
            throw new Error("this browser cannot suspend WebAssembly (JavaScript Promise Integration)");
          },
  };
}

/// The one Orla a page loads, by the URL of the glue `wasm-bindgen`
/// wrote for it (`orla_ffi.js`, beside `orla_ffi_bg.wasm`). Its imports
/// from `orla` are `orla-host.mjs`, which an import map names.
const loaded = new Map();

/// Orla itself in the page: the session as WebAssembly (the C ABI,
/// `orla_ffi.wasm` built for `wasm32-unknown-unknown`), its documents the
/// files handed in, no disk and no server. It answers the same protocol
/// lines a socket carries, through `orla_request`, so a Driver runs it as
/// it runs a connection. A call that waits for the device suspends Orla
/// and answers when the device has; each session is its own, over the
/// one instance the page loaded.
class OrlaInPage {
  constructor(exports) {
    this.x = exports;
    // What may wait for a device answers a promise.
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(exports[name]) : exports[name];
    this.open_files = promising("orla_open_files");
    this.open_replay = promising("orla_open_replay");
    this.request = promising("orla_request");
    this.presenting = promising("orla_present");
    this.files = exports.orla_files();
    this.session = 0;
    this.onclose = null;
    // One call at a time: a call waiting on the device has Orla
    // suspended, and another reaching into it then would find a session
    // half through its frame.
    this.line = Promise.resolve();
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
  }

  /// Loads Orla from its glue's URL, once a page, and makes a new set of
  /// files for a session of its own.
  static async load(glue) {
    const url = new URL(glue, location.href).href;
    if (!loaded.has(url)) {
      loaded.set(
        url,
        (async () => {
          const [module, hosted] = await Promise.all([import(url), import("orla")]);
          const exports = await module.default();
          hosted.bind({ exports });
          return exports;
        })(),
      );
    }
    return new OrlaInPage(await loaded.get(url));
  }

  memory() {
    return new Uint8Array(this.x.memory.buffer);
  }

  /// `bytes` copied into Orla's memory for one call, which may answer a
  /// promise: the room is given back when it settles.
  withBytes(bytes, use) {
    const pointer = this.x.orla_alloc(bytes.length);
    if (!pointer) throw new Error("Orla has no room for the call");
    this.memory().set(bytes, pointer);
    const give = () => this.x.orla_free(pointer, bytes.length);
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

  /// Puts a file at `path` (`examples/text/scene.osn`), replacing what
  /// was there; an open session reads the change when asked to `sync`,
  /// as it reads a file changed on disk.
  put(path, bytes) {
    this.withText(path, (name) =>
      this.withBytes(bytes, (data, length) => this.x.orla_files_put(this.files, name, data, length)),
    );
  }

  /// Opens `scene`, a path among the files put: on the JIT, whose code
  /// this page loads, or on `gpu`, the browser's WebGPU, or `cpu`.
  async open(scene, backend = "jit") {
    if (this.session) this.x.orla_close(this.session);
    this.session = await this.withText(scene, (name) =>
      this.withText(backend, (on) => this.open_files(this.files, name, on)),
    );
    if (!this.session) throw new Error(this.text(this.x.orla_open_error()));
  }

  /// Opens the scene the replay at `path` (among the files put) names,
  /// and plays it: see `OrlaPlayer`.
  async openReplay(path, backend = "jit") {
    if (this.session) this.x.orla_close(this.session);
    this.session = await this.withText(path, (name) =>
      this.withText(backend, (on) => this.open_replay(this.files, name, on)),
    );
    if (!this.session) throw new Error(this.text(this.x.orla_open_error()));
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
    if (done < 0) throw new Error(this.text(this.x.orla_error(this.session)));
    return done === 0;
  }

  close() {
    if (this.session) this.x.orla_close(this.session);
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
/// `targetHeight` canvas, aspect kept, centred: the Player's placement.
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
/// pointer, buttons, wheel and keys over it are the input, and each frame
/// is an `evaluate` followed by a `read` drawn onto the canvas. Given
/// `shown`, a second canvas over the first that takes no pointer, a
/// session that can present (Orla in the page, on its GPU) draws each
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

  /// The key codes a Kernel reads: letters and digits by uppercase ASCII,
  /// arrows 37 to 40, space 32, enter 13, shift 16, escape 27.
  static keyCode(event) {
    if (event.key.length === 1) {
      const upper = event.key.toUpperCase();
      return upper.charCodeAt(0);
    }
    return { ArrowLeft: 37, ArrowUp: 38, ArrowRight: 39, ArrowDown: 40, Enter: 13, Shift: 16, Escape: 27, Tab: 9, Backspace: 8 }[event.key];
  }

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
      const code = Driver.keyCode(event);
      if (code !== undefined) {
        this.keys.add(code);
        event.preventDefault();
      }
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
    const started = performance.now();
    const report = await this.client.call("evaluate", {
      time: (started - this.started) / 1000,
      pointer: this.pointer,
      buttons: [...this.buttons],
      wheel,
      keys: [...this.keys],
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

globalThis.Orla = { OrlaClient, OrlaInPage, host, Driver, decodePpm, decodeWords, placement, bytesOf };
})();
