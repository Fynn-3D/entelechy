// Entelechy's single-file page player: boots Entelechy's slim bundle runtime
// (`entelechy_single`), decodes and prepares the embedded binary `Bundle` in
// turns, and drives the shared play (`entelechy_play_*`) over the page's
// canvas, WebGPU surface, WebAudio worklet, pointer, keyboard and gamepads.

(() => {
const PAD_BUTTONS = [
  "south", "east", "west", "north", "left_shoulder", "right_shoulder", null, null,
  "select", "start", "left_thumb", "right_thumb", "up", "down", "left", "right",
];
const BUTTON = { 0: 0, 1: 2, 2: 1 };
const BLOCKS_AHEAD = 4;
const AHEAD = "ahead";
const RESTING = "resting";
const IS_LE = new Uint8Array(new Uint16Array([1]).buffer)[0] === 1;
const HELD = new Set([
  "Tab", "Escape", "F5", "F6", "F7", "F8", "F9", "F10",
  "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ", "Space",
]);
const CODE_KEYS = {
  Space: "Space", Enter: "Enter", NumpadEnter: "Enter", Escape: "Escape",
  Backspace: "Backspace", Tab: "Tab",
  ShiftLeft: "Shift", ShiftRight: "Shift",
  ControlLeft: "Control", ControlRight: "Control",
  AltLeft: "Alt", AltRight: "Alt",
  MetaLeft: "Meta", MetaRight: "Meta",
  PageUp: "PageUp", PageDown: "PageDown", End: "End", Home: "Home",
  ArrowLeft: "ArrowLeft", ArrowUp: "ArrowUp", ArrowRight: "ArrowRight", ArrowDown: "ArrowDown",
  Delete: "Delete",
  Semicolon: ";", Equal: "=", Comma: ",", Minus: "-", Period: ".", Slash: "/",
  Backquote: "`", BracketLeft: "[", Backslash: "\\", BracketRight: "]", Quote: "'",
};
let rawPointerSupported = true;

function codeKey(event) {
  const code = event.code || "";
  if (code.length === 4 && code.startsWith("Key")) return code.slice(3);
  if (code.length === 6 && code.startsWith("Digit")) return code.slice(5);
  if (code.length === 7 && code.startsWith("Numpad")) {
    const d = code.charCodeAt(6);
    if (d >= 48 && d <= 57) return code.slice(6);
  }
  if (/^F([1-9]|1[0-2])$/.test(code)) return code;
  return CODE_KEYS[code] || event.key || "";
}

function lockPointer(canvas, onRefused) {
  if (typeof canvas.requestPointerLock !== "function") {
    onRefused();
    return;
  }
  const attempt = (raw) => {
    try {
      const asked = raw ? canvas.requestPointerLock({ unadjustedMovement: true }) : canvas.requestPointerLock();
      if (asked && typeof asked.catch === "function") {
        asked.catch((error) => {
          if (raw) {
            if (error && error.name === "NotSupportedError") rawPointerSupported = false;
            attempt(false);
          } else {
            onRefused();
          }
        });
      }
    } catch {
      if (raw) {
        rawPointerSupported = false;
        attempt(false);
      } else {
        onRefused();
      }
    }
  };
  attempt(rawPointerSupported);
}

const WORKLET = `
class EntelechySound extends AudioWorkletProcessor {
  constructor() {
    super();
    this.blocks = [];
    this.at = 0;
    this.port.onmessage = (event) => this.blocks.push(event.data);
  }
  process(inputs, outputs) {
    const out = outputs[0];
    const frames = out[0].length;
    for (let i = 0; i < frames; i++) {
      let left = 0, right = 0;
      const block = this.blocks[0];
      if (block) {
        left = block.samples[this.at * block.channels];
        right = block.channels === 2 ? block.samples[this.at * 2 + 1] : left;
        if (++this.at * block.channels >= block.samples.length) { this.blocks.shift(); this.at = 0; }
      }
      out[0][i] = left;
      if (out[1]) out[1][i] = right;
    }
    let held = 0;
    for (const block of this.blocks) held += block.samples.length / block.channels;
    this.port.postMessage(held - this.at);
    return true;
  }
}
registerProcessor("entelechy-sound", EntelechySound);
`;
const WORKLET_URL = `data:text/javascript;charset=utf-8,${encodeURIComponent(WORKLET)}`;

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

function makeHost() {
  let exports = null;
  const free = [];
  const parked = parking();
  const table = () => exports.__indirect_function_table;
  return {
    bind(bound) {
      exports = bound;
    },
    load(bytes, length, frame, slots, most) {
      if (!length) return typeof WebAssembly.Module === "function" ? 0 : -2;
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

function wear(canvas, look) {
  if (look === null) return;
  if (look === "system") {
    canvas.style.cursor = "";
    return;
  }
  const drawn = document.createElement("canvas");
  drawn.width = look.width;
  drawn.height = look.height;
  drawn.getContext("2d").putImageData(new ImageData(look.rgba, look.width, look.height), 0, 0);
  canvas.style.cursor = `url(${drawn.toDataURL()}) ${look.hotspot[0]} ${look.hotspot[1]}, auto`;
}

function written(event) {
  return event.key.length === 1 && !event.ctrlKey && !event.metaKey && !event.isComposing ? event.key : "";
}

async function gzipPipe(bytes, decompress) {
  if (typeof CompressionStream !== "function" || typeof DecompressionStream !== "function") return null;
  try {
    const stream = new Blob([bytes])
      .stream()
      .pipeThrough(decompress ? new DecompressionStream("gzip") : new CompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  } catch {
    return null;
  }
}

async function allocAndOpenBundle(exports, bundleBytes, backendName) {
  const owned = typeof exports.entelechy_open_bundle_owned === "function";
  const rawOpen = owned ? exports.entelechy_open_bundle_owned : exports.entelechy_open_bundle;
  const openBundle =
    typeof WebAssembly.promising === "function" ? WebAssembly.promising(rawOpen) : rawOpen;
  const enc = new TextEncoder();
  const dec = new TextDecoder();
  const readText = (ptr) => {
    const mem = new Uint8Array(exports.memory.buffer);
    let end = ptr;
    while (mem[end]) end++;
    return dec.decode(mem.subarray(ptr, end));
  };
  const backendBytes = enc.encode(`${backendName || "best"}\0`);
  const bundleLen = bundleBytes.length;
  const dataPtr = exports.entelechy_alloc(bundleLen);
  const backPtr = exports.entelechy_alloc(backendBytes.length);
  if (!dataPtr || !backPtr) {
    if (dataPtr) exports.entelechy_free(dataPtr, bundleLen);
    if (backPtr) exports.entelechy_free(backPtr, backendBytes.length);
    throw new Error("Entelechy has no room for the bundle");
  }
  let session = 0;
  let passedOwnership = false;
  try {
    new Uint8Array(exports.memory.buffer, dataPtr, bundleLen).set(bundleBytes);
    bundleBytes = null;
    new Uint8Array(exports.memory.buffer, backPtr, backendBytes.length).set(backendBytes);
    passedOwnership = owned;
    session = await openBundle(dataPtr, bundleLen, backPtr);
  } finally {
    if (!passedOwnership) exports.entelechy_free(dataPtr, bundleLen);
    exports.entelechy_free(backPtr, backendBytes.length);
  }
  if (!session) throw new Error(readText(exports.entelechy_open_error()));
  return session;
}

class SinglePlayer {
  constructor(exports, session, canvas, shown, pointer = "free", bundleGz = null, needsGpu = false, onfail = null) {
    this.x = exports;
    this.session = session;
    this.canvas = canvas;
    this.shown = shown;
    this.pointer = pointer || "free";
    this._bundleGz = bundleGz;
    this.needsGpu = !!needsGpu;
    this.onfail = typeof onfail === "function" ? onfail : null;
    this.shownFrames = 0;
    this.context = canvas.getContext("2d");
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(exports[name]) : exports[name];
    this.turn = promising("entelechy_play_turn");
    this.frame = promising("entelechy_play_frame");
    this.present = promising("entelechy_play_present");
    this.pixels = promising("entelechy_play_pixels");
    this.sound = promising("entelechy_play_sound");
    this.flushSaves =
      typeof exports.entelechy_save_flush === "function" ? promising("entelechy_save_flush") : null;
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
    this.line = Promise.resolve();
    this.running = false;
    this.pads = new Map();
    this.audio = null;
    this.startingSound = null;
    this.held = 0;
    this.ring = 0;
    this.said = [];
    this.onsaid = (line) => console.log(line);
    this.hands = [];
    this.wakeWaiter = null;
    this.downByCode = new Map();
    this.heldAt = [0, 0];
    this.swallowedButton = null;
    this.lockFailedOnce = false;
    this.saveSlots = [];
    this.savePrefix = `entelechy.save:${location.pathname}:${document.title}:`;
    this.initSaves();
    this.listen();
  }

  static async start({ glue, wasm, bundle, bundleGzip = null, backend, pointer, canvas, shown, told = () => {}, onfail = null }) {
    globalThis.entelechy_webgl_canvas = shown || null;
    globalThis.entelechy_webgl ??= makeWebGlHost();
    const host = makeHost();
    globalThis.__entelechy_host = host;
    const hostUrl = URL.createObjectURL(
      new Blob(
        ["const h = globalThis.__entelechy_host;\nexport const { load, unload, wait, wake, now } = h;\n"],
        { type: "text/javascript" },
      ),
    );
    const patched = glue
      .replaceAll('from "entelechy"', `from ${JSON.stringify(hostUrl)}`)
      .replaceAll("from 'entelechy'", `from ${JSON.stringify(hostUrl)}`);
    glue = null;
    const glueUrl = URL.createObjectURL(new Blob([patched], { type: "text/javascript" }));
    await told(0.2, "starting Entelechy");
    let exports;
    try {
      const mod = await import(glueUrl);
      exports = await mod.default({ module_or_path: wasm });
      wasm = null;
      host.bind(exports);
    } finally {
      URL.revokeObjectURL(glueUrl);
      URL.revokeObjectURL(hostUrl);
    }

    await told(0.35, "opening scene");
    const chosenBackend = backend || "best";
    const bundleGz = chosenBackend === "jit" ? null : (bundleGzip || (await gzipPipe(bundle, false)));
    const session = await allocAndOpenBundle(exports, bundle, chosenBackend);
    bundle = null;

    const needsGpu =
      typeof exports.entelechy_play_needs_gpu === "function" && exports.entelechy_play_needs_gpu(session) === 1;
    const hasComputeGpu =
      typeof exports.entelechy_play_compute_gpu !== "function" || exports.entelechy_play_compute_gpu(session) === 1;
    if (
      needsGpu &&
      !hasComputeGpu &&
      chosenBackend !== "jit" &&
      !globalThis.entelechy_webgl_only &&
      !globalThis.entelechy_force_fallback
    ) {
      try { exports.entelechy_close(session); } catch {}
      throw new Error(
        "WebGPU compute is unavailable on this browser or graphics driver; this 3D world is too heavy for single-threaded CPU fallback (add ?on=jit to force CPU mode)"
      );
    }

    const player = new SinglePlayer(
      exports,
      session,
      canvas,
      shown,
      pointer,
      needsGpu ? null : bundleGz,
      needsGpu,
      onfail,
    );
    let left = exports.entelechy_play_preparing(session);
    let most = left;
    let high = 0.4;
    while (left > 0) {
      most = Math.max(most, left);
      high = Math.max(high, 0.4 + 0.52 * (1 - left / most));
      await told(high, `preparing ${most - left} of ${most}`);
      left = await player.inTurn(() => player.turn(session));
      player.say();
      if (left < 0) throw new Error(player.text(exports.entelechy_error(session)));
    }
    player.say();
    if (player.canvas.clientWidth > 0 && player.canvas.clientHeight > 0) {
      await told(0.95, "building first frame");
      for (let guard = 0; guard < 16 && player.shownFrames === 0 && !player.failed; guard++) {
        await player.step();
      }
      if (player.failed) {
        const msg = player.failed;
        try { exports.entelechy_close(player.session); } catch {}
        player.session = 0;
        throw new Error(msg);
      }
    }
    return player;
  }

  memory() {
    return new Uint8Array(this.x.memory.buffer);
  }

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

  inTurn(work) {
    const done = this.line.then(work);
    this.line = done.catch(() => {});
    return done;
  }

  typed(text) {
    this.withText(text, (at) => this.x.entelechy_play_typed(this.session, at));
  }

  key(name, down) {
    this.withText(name, (at) => this.x.entelechy_play_key(this.session, at, down ? 1 : 0));
  }

  pointerLook() {
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

  wake() {
    if (this.wakeWaiter) {
      const done = this.wakeWaiter;
      this.wakeWaiter = null;
      done();
    }
  }

  hand(fn) {
    this.hands.push(fn);
    this.wake();
  }

  waitForWake() {
    if (this.hands.length > 0 || !this.running) return Promise.resolve();
    return new Promise((done) => {
      this.wakeWaiter = done;
    });
  }

  hasConnectedPad() {
    if (!navigator.getGamepads) return false;
    for (const pad of navigator.getGamepads()) {
      if (pad && pad.mapping === "standard") return true;
    }
    return false;
  }

  releaseAll() {
    this.downByCode.clear();
    this.swallowedButton = null;
    this.hand(() => {
      if (this.x.entelechy_play_release) this.x.entelechy_play_release(this.session);
    });
  }

  listen() {
    window.addEventListener("gamepadconnected", () => this.wake());
    window.addEventListener("keydown", (event) => {
      const text = written(event);
      if (text) this.hand(() => this.typed(text));
      const key = codeKey(event);
      if (HELD.has(event.key) || HELD.has(key)) event.preventDefault();
      if (event.repeat || !key) return;
      if (event.code) this.downByCode.set(event.code, key);
      this.startSound().catch(() => {});
      this.hand(() => this.key(key, true));
      if (!this.running) this.run();
    });
    window.addEventListener("keyup", (event) => {
      const key = (event.code && this.downByCode.get(event.code)) || codeKey(event);
      if (event.code) this.downByCode.delete(event.code);
      if (HELD.has(event.key) || HELD.has(key)) event.preventDefault();
      if (!key) return;
      this.hand(() => this.key(key, false));
    });
    window.addEventListener("blur", () => this.releaseAll());
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) {
        this.releaseAll();
        if (this.saveSlots.length && this.flushSaves) {
          this.inTurn(async () => {
            await this.flushSaves(this.session);
            this.syncSaves();
          });
        }
      } else {
        this.wake();
      }
    });
    window.addEventListener("pagehide", () => {
      this.syncSaves();
    });
    document.addEventListener("pointerlockchange", () => {
      const locked = document.pointerLockElement === this.canvas;
      document.body.classList.toggle("locked", locked);
      if (locked) this.lockFailedOnce = false;
      else this.releaseAll();
    });
    window.addEventListener("pointerdown", () => {
      this.startSound().catch(() => {});
    });
    const scale = () => window.devicePixelRatio || 1;
    const clampHeld = (dx, dy) => {
      const mostX = Math.max(0, this.canvas.width - 1);
      const mostY = Math.max(0, this.canvas.height - 1);
      this.heldAt = [
        Math.min(mostX, Math.max(0, this.heldAt[0] + dx)),
        Math.min(mostY, Math.max(0, this.heldAt[1] + dy)),
      ];
      return this.heldAt;
    };
    this.canvas.addEventListener("pointermove", (event) => {
      const s = scale();
      if (document.pointerLockElement === this.canvas) {
        const [dx, dy] = [event.movementX * s, event.movementY * s];
        if (this.pointer === "lock" && this.x.entelechy_play_pointer_moved) {
          this.hand(() => this.x.entelechy_play_pointer_moved(this.session, dx, dy));
        } else {
          const [x, y] = clampHeld(dx, dy);
          this.hand(() => this.x.entelechy_play_pointer(this.session, x, y));
        }
        return;
      }
      if (this.pointer === "lock") {
        if (event.buttons !== 0 && this.x.entelechy_play_pointer_moved) {
          const [dx, dy] = [event.movementX * s, event.movementY * s];
          this.hand(() => this.x.entelechy_play_pointer_moved(this.session, dx, dy));
        }
        return;
      }
      const [x, y] = [event.offsetX * s, event.offsetY * s];
      this.heldAt = [x, y];
      this.hand(() => this.x.entelechy_play_pointer(this.session, x, y));
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      this.canvas.focus();
      event.preventDefault();
      this.startSound().catch(() => {});
      const s = scale();
      this.heldAt = [event.offsetX * s, event.offsetY * s];
      const button = BUTTON[event.button] ?? 0;
      if ((this.pointer === "lock" || this.pointer === "hold") && document.pointerLockElement !== this.canvas) {
        const swallow = this.pointer === "lock" && !this.lockFailedOnce;
        lockPointer(this.canvas, () => {
          this.lockFailedOnce = true;
        });
        if (swallow) {
          this.swallowedButton = button;
          if (!this.running) this.run();
          return;
        }
      }
      if (document.pointerLockElement !== this.canvas && this.canvas.setPointerCapture) {
        try {
          this.canvas.setPointerCapture(event.pointerId);
        } catch {}
      }
      this.hand(() => this.x.entelechy_play_button(this.session, button, 1));
      if (!this.running) this.run();
    });
    this.canvas.addEventListener("pointerup", (event) => {
      const button = BUTTON[event.button] ?? 0;
      if (this.swallowedButton === button) {
        this.swallowedButton = null;
        return;
      }
      this.hand(() => this.x.entelechy_play_button(this.session, button, 0));
    });
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 1 : 1 / 40;
      const [x, y] = [event.deltaX * unit, -event.deltaY * unit];
      this.hand(() => this.x.entelechy_play_wheel(this.session, x, y));
    }, { passive: false });
    const fit = () => {
      const width = Math.max(1, Math.round(this.canvas.clientWidth * scale()));
      const height = Math.max(1, Math.round(this.canvas.clientHeight * scale()));
      let changed = false;
      for (const canvas of [this.canvas, this.shown]) {
        if (canvas.width !== width || canvas.height !== height) {
          canvas.width = width;
          canvas.height = height;
          changed = true;
        }
      }
      if (changed) this.wake();
    };
    new ResizeObserver(fit).observe(this.canvas);
    fit();
  }

  pollPads() {
    let connected = 0;
    for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
      if (!pad || pad.mapping !== "standard") continue;
      connected += 1;
      const last = this.pads.get(pad.index) || { buttons: [], axes: [] };
      const axis = (name, value, at) => {
        if (last.axes[at] !== value) {
          this.withText(name, (text) => this.x.entelechy_play_axis(this.session, text, value));
        }
      };
      pad.buttons.forEach((button, at) => {
        const name = PAD_BUTTONS[at];
        if (name && last.buttons[at] !== button.pressed) {
          this.withText(name, (text) => this.x.entelechy_play_pad(this.session, text, button.pressed ? 1 : 0));
        }
      });
      axis("left_x", pad.axes[0] ?? 0, 0);
      axis("left_y", -(pad.axes[1] ?? 0), 1);
      axis("right_x", pad.axes[2] ?? 0, 2);
      axis("right_y", -(pad.axes[3] ?? 0), 3);
      axis("left_trigger", pad.buttons[6]?.value ?? 0, 4);
      axis("right_trigger", pad.buttons[7]?.value ?? 0, 5);
      this.pads.set(pad.index, {
        buttons: pad.buttons.map((button) => button.pressed),
        axes: [
          pad.axes[0] ?? 0, -(pad.axes[1] ?? 0),
          pad.axes[2] ?? 0, -(pad.axes[3] ?? 0),
          pad.buttons[6]?.value ?? 0, pad.buttons[7]?.value ?? 0,
        ],
      });
    }
    this.x.entelechy_play_pads(this.session, connected);
  }

  initSaves() {
    if (typeof this.x.entelechy_save_slot !== "function" || typeof this.x.entelechy_save_load !== "function") {
      return;
    }
    for (let i = 0; ; i++) {
      const ptr = this.x.entelechy_save_slot(this.session, i);
      if (!ptr) break;
      this.saveSlots.push(this.text(ptr));
    }
    if (!this.saveSlots.length) return;
    for (const slot of this.saveSlots) {
      let b64 = null;
      try {
        b64 = globalThis.localStorage ? localStorage.getItem(this.savePrefix + slot) : null;
      } catch {}
      if (!b64) continue;
      try {
        const raw = atob(b64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        this.withText(slot, (slotPtr) =>
          this.withBytes(bytes, (dataPtr, dataLen) =>
            this.x.entelechy_save_load(this.session, slotPtr, dataPtr, dataLen)
          )
        );
      } catch {}
    }
  }

  syncSaves() {
    if (!this.saveSlots.length || typeof this.x.entelechy_save_take !== "function") return;
    const lenPtr = this.x.entelechy_alloc(4);
    if (!lenPtr) return;
    try {
      for (let i = 0; i < this.saveSlots.length; i++) {
        const dataPtr = this.x.entelechy_save_take(this.session, i, lenPtr);
        if (!dataPtr) continue;
        const len = new Uint32Array(this.x.memory.buffer, lenPtr, 1)[0];
        const bytes = new Uint8Array(this.x.memory.buffer, dataPtr, len);
        let bin = "";
        for (let off = 0; off < bytes.length; off += 0x8000) {
          bin += String.fromCharCode(...bytes.subarray(off, off + 0x8000));
        }
        try {
          if (globalThis.localStorage) localStorage.setItem(this.savePrefix + this.saveSlots[i], btoa(bin));
        } catch {}
      }
    } finally {
      this.x.entelechy_free(lenPtr, 4);
    }
  }

  say() {
    const said = this.text(this.x.entelechy_play_said(this.session));
    for (const line of said.split("\n").filter(Boolean)) {
      this.said.push(line);
      this.onsaid(line);
    }
  }

  step() {
    return this.inTurn(() => this.stepNow());
  }

  async stepNow() {
    const hands = this.hands;
    this.hands = [];
    for (const hand of hands) hand();
    this.pollPads();
    this.say();
    if (this.audio && this.ring && this.held + this.audio.context.sampleRate / 60 > this.ring) return AHEAD;
    const rate = this.audio ? this.audio.context.sampleRate : 0;
    const scale = window.devicePixelRatio || 1;
    this.x.entelechy_play_sound_queued(this.session, this.audio ? this.held : 0);
    const framed = await this.frame(this.session, this.canvas.width, this.canvas.height, scale, rate);
    this.say();
    if (framed === 0) {
      this.shownFrames++;
      this._bundleGz = null;
      this.failed = null;
      this.syncSaves();
      await this.play();
      await this.show();
      wear(this.canvas, this.pointerLook());
      if (
        typeof this.x.entelechy_play_resting === "function" &&
        this.x.entelechy_play_resting(this.session) === 1 &&
        this.hands.length === 0 &&
        !this.hasConnectedPad()
      ) {
        return RESTING;
      }
    } else if (framed < 0) {
      const error = this.text(this.x.entelechy_error(this.session));
      if (
        !this.shownFrames &&
        this._bundleGz &&
        /device (was )?lost|device failed|lost the device/i.test(error) &&
        (await this.recoverFromDeviceLoss(error))
      ) {
        return;
      }
      document.title = `Entelechy: ${error}`;
      if (error !== this.failed) {
        console.error(`Entelechy: ${error}`);
        if (this.onfail) {
          try { this.onfail(error); } catch {}
        }
      }
      this.failed = error;
    }
  }

  async recoverFromDeviceLoss(why) {
    if (this.needsGpu && !globalThis.entelechy_force_fallback) {
      this._bundleGz = null;
      return false;
    }
    const gz = this._bundleGz;
    this._bundleGz = null;
    if (!gz) return false;
    const candidates = globalThis.entelechy_webgl_only ? ["jit"] : ["webgl2", "jit"];
    for (const mode of candidates) {
      try {
        const raw = await gzipPipe(gz, true);
        if (!raw) return false;
        if (this.session) {
          try { this.x.entelechy_close(this.session); } catch {}
          this.session = 0;
        }
        if (mode === "webgl2") globalThis.entelechy_webgl_only = true;
        const nextBackend = mode === "webgl2" ? "gpu" : "jit";
        this.onsaid(`GPU lost (${why.slice(0, 80)}), falling back to ${mode}…`);
        const nextSession = await allocAndOpenBundle(this.x, raw, nextBackend);
        this.session = nextSession;
        let left = this.x.entelechy_play_preparing(nextSession);
        while (left > 0) {
          left = await this.turn(nextSession);
          this.say();
          if (left < 0) throw new Error(this.text(this.x.entelechy_error(nextSession)));
        }
        this.initSaves();
        if (this.audio) {
          const described = `${this.audio.context.sampleRate} Hz`;
          this.withText(described, (text) => this.x.entelechy_play_sound_opened(this.session, text));
        }
        return true;
      } catch (err) {
        this.onsaid(`${mode} fallback refused: ${err.message}`);
      }
    }
    return false;
  }

  async show() {
    const presented = await this.withText(this.shown.id, (id) => this.present(this.session, id));
    this.shown.style.display = presented === 0 ? "block" : "none";
    if (presented === 0) return;
    const { width, height } = this.canvas;
    if (!width || !height) return;
    const pointer = await this.pixels(this.session, width, height);
    if (!pointer) return;
    const len = width * height;
    const words = new Uint32Array(this.x.memory.buffer, pointer, len);
    if (!this._imgData || this._imgW !== width || this._imgH !== height) {
      this._imgData = this.context.createImageData(width, height);
      this._imgU32 = new Uint32Array(this._imgData.data.buffer);
      this._imgW = width;
      this._imgH = height;
    }
    if (IS_LE) {
      const out32 = this._imgU32;
      for (let i = 0; i < len; i++) {
        const w = words[i];
        out32[i] = 0xff000000 | ((w & 255) << 16) | (w & 0xff00) | ((w >> 16) & 255);
      }
    } else {
      const data = this._imgData.data;
      for (let i = 0; i < len; i++) {
        const w = words[i];
        const j = i << 2;
        data[j] = (w >> 16) & 255;
        data[j + 1] = (w >> 8) & 255;
        data[j + 2] = w & 255;
        data[j + 3] = 255;
      }
    }
    this.context.putImageData(this._imgData, 0, 0);
  }

  async play() {
    if (!this.audio || !this.audio.node) return;
    const field = this.x.entelechy_alloc(36);
    try {
      if ((await this.sound(this.session, field)) !== 0) return;
      const at = new Uint32Array(this.x.memory.buffer, field, 9);
      const [words, , , , count, stride] = at;
      const samples = new Float32Array(new Float32Array(this.x.memory.buffer, words, count * stride));
      if (!this.ring) this.ring = Math.max(count, Math.floor(this.audio.context.sampleRate / 30), 64) * BLOCKS_AHEAD;
      this.audio.node.port.postMessage({ samples, channels: stride }, [samples.buffer]);
    } finally {
      this.x.entelechy_free(field, 36);
    }
  }

  async startSound() {
    if (this.audio) {
      if (this.audio.context.state === "suspended") await this.audio.context.resume();
      return;
    }
    if (this.startingSound) return this.startingSound;
    this.startingSound = (async () => {
      const context = new AudioContext();
      try {
        await context.audioWorklet.addModule(WORKLET_URL);
        if (context.state === "suspended") await context.resume();
        const node = new AudioWorkletNode(context, "entelechy-sound", { outputChannelCount: [2] });
        node.port.onmessage = (event) => (this.held = event.data);
        node.connect(context.destination);
        this.audio = { context, node };
        const described = `${context.sampleRate} Hz`;
        this.hand(() => this.withText(described, (text) => this.x.entelechy_play_sound_opened(this.session, text)));
      } catch (error) {
        await context.close().catch(() => {});
        throw error;
      } finally {
        this.startingSound = null;
      }
    })();
    return this.startingSound;
  }

  async run() {
    if (this.running) return;
    this.running = true;
    while (this.running) {
      let stepped;
      try {
        stepped = await this.step();
      } catch (error) {
        this.onsaid(`stopped: ${error.message}`);
        this.running = false;
        return;
      }
      if (stepped === RESTING) {
        await this.waitForWake();
        if (!this.running) break;
        await new Promise((done) => requestAnimationFrame(done));
      } else {
        await new Promise((done) => (stepped === AHEAD ? setTimeout(done, 1) : requestAnimationFrame(done)));
      }
    }
  }

  stop() {
    this.running = false;
    this.wake();
  }
}

globalThis.EntelechySingle = SinglePlayer;

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
    if (typeof document !== "undefined" && name) return document.getElementById(name);
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
      if (gl && !gl.isContextLost() && (!desired || gl.canvas === desired)) return info;
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
        let ctx = desired && typeof desired.getContext === "function" ? desired.getContext("webgl2", opts) : null;
        if (!ctx) {
          const fb = typeof OffscreenCanvas !== "undefined" ? new OffscreenCanvas(1, 1) : document.createElement("canvas");
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
