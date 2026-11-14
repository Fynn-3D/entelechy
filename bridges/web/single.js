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
const HELD = new Set([
  "Tab", "Escape", "F5", "F6", "F7", "F8", "F9", "F10",
  "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " ",
]);

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

function makeHost() {
  let exports = null;
  const free = [];
  const parked = parking();
  const table = () => exports.__indirect_function_table;
  return {
    bind(bound) {
      exports = bound;
    },
    load(bytes, length, slots, most) {
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
  return event.key.length === 1 && !event.ctrlKey && !event.metaKey ? event.key : "";
}

class SinglePlayer {
  constructor(exports, session, canvas, shown) {
    this.x = exports;
    this.session = session;
    this.canvas = canvas;
    this.shown = shown;
    this.context = canvas.getContext("2d");
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(exports[name]) : exports[name];
    this.turn = promising("entelechy_play_turn");
    this.frame = promising("entelechy_play_frame");
    this.present = promising("entelechy_play_present");
    this.pixels = promising("entelechy_play_pixels");
    this.sound = promising("entelechy_play_sound");
    this.encoder = new TextEncoder();
    this.decoder = new TextDecoder();
    this.line = Promise.resolve();
    this.running = false;
    this.pads = new Map();
    this.audio = null;
    this.held = 0;
    this.ring = 0;
    this.said = [];
    this.onsaid = (line) => console.log(line);
    this.hands = [];
    this.listen();
  }

  static async start({ glue, wasm, bundle, backend, canvas, shown, told }) {
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
    const backendBytes = enc.encode(`${backend || "best"}\0`);
    const bundleLen = bundle.length;
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
      new Uint8Array(exports.memory.buffer, dataPtr, bundleLen).set(bundle);
      bundle = null;
      new Uint8Array(exports.memory.buffer, backPtr, backendBytes.length).set(backendBytes);
      passedOwnership = owned;
      session = await openBundle(dataPtr, bundleLen, backPtr);
    } finally {
      if (!passedOwnership) exports.entelechy_free(dataPtr, bundleLen);
      exports.entelechy_free(backPtr, backendBytes.length);
    }
    if (!session) throw new Error(readText(exports.entelechy_open_error()));

    const player = new SinglePlayer(exports, session, canvas, shown);
    let left = exports.entelechy_play_preparing(session);
    let most = left;
    while (left > 0) {
      most = Math.max(most, left);
      await told(0.4 + 0.6 * (1 - left / most), `preparing ${most - left} of ${most}`);
      left = await player.inTurn(() => player.turn(session));
      player.say();
      if (left < 0) throw new Error(player.text(exports.entelechy_error(session)));
    }
    player.say();
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

  listen() {
    window.addEventListener("keydown", (event) => {
      const text = written(event);
      if (text) this.hands.push(() => this.typed(text));
      if (event.repeat) return;
      if (HELD.has(event.key)) event.preventDefault();
      this.startSound().catch(() => {});
      const key = event.key;
      this.hands.push(() => this.key(key, true));
      if (!this.running) this.run();
    });
    window.addEventListener("keyup", (event) => {
      if (HELD.has(event.key)) event.preventDefault();
      const key = event.key;
      this.hands.push(() => this.key(key, false));
    });
    const scale = () => window.devicePixelRatio || 1;
    this.canvas.addEventListener("pointermove", (event) => {
      const [x, y] = [event.offsetX * scale(), event.offsetY * scale()];
      this.hands.push(() => this.x.entelechy_play_pointer(this.session, x, y));
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      this.startSound().catch(() => {});
      this.canvas.setPointerCapture(event.pointerId);
      const button = BUTTON[event.button] ?? 0;
      this.hands.push(() => this.x.entelechy_play_button(this.session, button, 1));
      if (!this.running) this.run();
    });
    this.canvas.addEventListener("pointerup", (event) => {
      const button = BUTTON[event.button] ?? 0;
      this.hands.push(() => this.x.entelechy_play_button(this.session, button, 0));
    });
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 1 : 1 / 40;
      const [x, y] = [event.deltaX * unit, -event.deltaY * unit];
      this.hands.push(() => this.x.entelechy_play_wheel(this.session, x, y));
    }, { passive: false });
    const fit = () => {
      const width = Math.max(1, Math.round(this.canvas.clientWidth * scale()));
      const height = Math.max(1, Math.round(this.canvas.clientHeight * scale()));
      for (const canvas of [this.canvas, this.shown]) {
        canvas.width = width;
        canvas.height = height;
      }
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
      await this.show();
      await this.play();
      wear(this.canvas, this.pointerLook());
    } else if (framed < 0) {
      const error = this.text(this.x.entelechy_error(this.session));
      document.title = `Entelechy: ${error}`;
      if (error !== this.failed) console.error(`Entelechy: ${error}`);
      this.failed = error;
    }
  }

  async show() {
    const presented = await this.withText(this.shown.id, (id) => this.present(this.session, id));
    this.shown.style.display = presented === 0 ? "block" : "none";
    if (presented === 0) return;
    const { width, height } = this.canvas;
    const pointer = await this.pixels(this.session, width, height);
    if (!pointer) return;
    const words = new Uint32Array(this.x.memory.buffer, pointer, width * height);
    const image = this.context.createImageData(width, height);
    for (let i = 0; i < words.length; i++) {
      const word = words[i];
      image.data[i * 4] = (word >> 16) & 255;
      image.data[i * 4 + 1] = (word >> 8) & 255;
      image.data[i * 4 + 2] = word & 255;
      image.data[i * 4 + 3] = 255;
    }
    this.context.putImageData(image, 0, 0);
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
    if (this.audio) return;
    const context = new AudioContext();
    this.audio = { context, node: null };
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const node = new AudioWorkletNode(context, "entelechy-sound", { outputChannelCount: [2] });
    node.port.onmessage = (event) => (this.held = event.data);
    node.connect(context.destination);
    this.audio.node = node;
    const described = `${context.sampleRate} Hz`;
    this.hands.push(() => this.withText(described, (text) => this.x.entelechy_play_sound_opened(this.session, text)));
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
      await new Promise((done) => (stepped === AHEAD ? setTimeout(done, 1) : requestAnimationFrame(done)));
    }
  }
}

globalThis.EntelechySingle = SinglePlayer;
})();
