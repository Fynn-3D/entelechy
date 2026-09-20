// Orla's creation protocol from a browser. A client of the public
// contract and nothing more: JSON lines as WebSocket text frames to a
// session serving with `--listen`, the same lines Blender or a shell
// would send on the socket. Nothing here knows how Orla is built.
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
/// is an `evaluate` followed by a `read` drawn onto the canvas.
class Driver {
  constructor(client, canvas) {
    this.client = client;
    this.canvas = canvas;
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
    if (picture) {
      const read = await this.client.call("read", { output: picture.name, format: "ppm" });
      const decoded = decodePpm(read.data);
      this.draw(decoded);
    }
    this.lastMillis = performance.now() - started;
    if (this.onframe) this.onframe(report, this.lastMillis);
    return report;
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

globalThis.Orla = { OrlaClient, Driver, decodePpm, decodeWords, placement, bytesOf };
})();
