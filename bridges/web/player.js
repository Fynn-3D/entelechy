// Orla's Player in a page: the page's devices for the play every
// interface shares (`orla_session::play`, reached through the C ABI's
// `orla_play_*`). What the keys mean, how a pad is a keyboard, what the
// line of frame data says, which output is the picture and which the
// sound, and how a replay drives the scene are the play's; this is only
// the canvas, the page's events, the Gamepad API, WebAudio, and the
// served files watched and handed in. A classic script after orla.js; it
// defines `OrlaPlayer`.

(() => {
const { OrlaInPage } = Orla;

/// The Gamepad API's standard mapping, button by button, in the play's
/// names; the triggers (6, 7) are axes.
const PAD_BUTTONS = [
  "south", "east", "west", "north", "left_shoulder", "right_shoulder", null, null,
  "select", "start", "left_thumb", "right_thumb", "up", "down", "left", "right",
];

/// A page's pointer button as the play's: 0 left, 1 right, 2 middle.
const BUTTON = { 0: 0, 1: 2, 2: 1 };

/// Keys the page would act on itself, which the scene has instead.
const HELD = new Set(["Tab", "F5", "F10", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " "]);

/// The sound card: a worklet that plays the blocks it is handed, in
/// order, and says how many samples it holds, which paces the frames as
/// the Player's ring does.
const WORKLET = `
class OrlaSound extends AudioWorkletProcessor {
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
registerProcessor("orla-sound", OrlaSound);
`;

class OrlaPlayer {
  /// Plays `orla` (an OrlaInPage with its scene open) on `canvas`, the
  /// page's window, with `shown` a canvas over it that takes no pointer,
  /// where the GPU presents.
  constructor(orla, canvas, shown) {
    this.orla = orla;
    this.x = orla.x;
    this.canvas = canvas;
    this.shown = shown;
    this.context = canvas.getContext("2d");
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(this.x[name]) : this.x[name];
    this.frame = promising("orla_play_frame");
    this.present = promising("orla_play_present");
    this.pixels = promising("orla_play_pixels");
    this.sound = promising("orla_play_sound");
    this.running = false;
    this.pads = new Map();
    this.audio = null;
    this.held = 0;
    this.block = 0;
    this.said = [];
    this.onsaid = (line) => console.log(line);
    this.listen();
  }

  /// Opens a scene or a replay (paths among `files`, [path, bytes]
  /// pairs), on `backend`, or as the Player does when none is named: the
  /// GPU, else the JIT, else the interpreter.
  static async open({ glue, files, scene, replay, backend, canvas, shown }) {
    const orla = await OrlaInPage.load(glue);
    for (const [path, bytes] of files) orla.put(path, bytes);
    const tried = backend ? [backend] : ["gpu", "jit", "cpu"];
    let failure = null;
    for (const on of tried) {
      try {
        if (replay) await orla.openReplay(replay, on);
        else await orla.open(scene, on);
        return new OrlaPlayer(orla, canvas, shown);
      } catch (error) {
        failure = error;
        console.log(`${on} backend unavailable (${error.message})`);
      }
    }
    throw failure;
  }

  get session() {
    return this.orla.session;
  }

  /// Hands a key to the play; true when it asks to stop.
  key(name, down) {
    return this.orla.withText(name, (at) => this.x.orla_play_key(this.session, at, down ? 1 : 0)) === 1;
  }

  listen() {
    window.addEventListener("keydown", (event) => {
      if (event.repeat) return;
      if (HELD.has(event.key)) event.preventDefault();
      this.startSound().catch(() => {});
      if (this.key(event.key, true)) this.stop();
      else if (!this.running && event.key !== "Escape") this.run();
      this.say();
    });
    window.addEventListener("keyup", (event) => {
      if (HELD.has(event.key)) event.preventDefault();
      this.key(event.key, false);
    });
    const scale = () => window.devicePixelRatio || 1;
    this.canvas.addEventListener("pointermove", (event) => {
      this.x.orla_play_pointer(this.session, event.offsetX * scale(), event.offsetY * scale());
    });
    this.canvas.addEventListener("pointerdown", (event) => {
      this.startSound().catch(() => {});
      this.canvas.setPointerCapture(event.pointerId);
      this.x.orla_play_button(this.session, BUTTON[event.button] ?? 0, 1);
      if (!this.running) this.run();
    });
    this.canvas.addEventListener("pointerup", (event) => {
      this.x.orla_play_button(this.session, BUTTON[event.button] ?? 0, 0);
    });
    this.canvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.canvas.addEventListener("wheel", (event) => {
      event.preventDefault();
      const unit = event.deltaMode === 1 ? 1 : 1 / 40;
      // Lines, y positive away from the person.
      this.x.orla_play_wheel(this.session, event.deltaX * unit, -event.deltaY * unit);
    }, { passive: false });
    const fit = () => {
      for (const canvas of [this.canvas, this.shown]) {
        canvas.width = Math.max(1, Math.round(canvas.clientWidth * scale()));
        canvas.height = Math.max(1, Math.round(canvas.clientHeight * scale()));
      }
    };
    new ResizeObserver(fit).observe(this.canvas);
    fit();
  }

  /// Hands the pads' buttons and axes to the play, where they changed.
  pollPads() {
    for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
      if (!pad || pad.mapping !== "standard") continue;
      const last = this.pads.get(pad.index) || { buttons: [], axes: [] };
      const axis = (name, value, at) => {
        if (last.axes[at] !== value) {
          this.orla.withText(name, (text) => this.x.orla_play_axis(this.session, text, value));
        }
      };
      pad.buttons.forEach((button, at) => {
        const name = PAD_BUTTONS[at];
        if (name && last.buttons[at] !== button.pressed) {
          this.orla.withText(name, (text) => this.x.orla_play_pad(this.session, text, button.pressed ? 1 : 0));
        }
      });
      // A page's stick reads down as positive; the play's up is.
      axis("left_x", pad.axes[0] ?? 0, 0);
      axis("left_y", -(pad.axes[1] ?? 0), 1);
      axis("right_x", pad.axes[2] ?? 0, 2);
      axis("right_y", -(pad.axes[3] ?? 0), 3);
      axis("left_trigger", pad.buttons[6]?.value ?? 0, 4);
      axis("right_trigger", pad.buttons[7]?.value ?? 0, 5);
      this.pads.set(pad.index, {
        buttons: pad.buttons.map((button) => button.pressed),
        axes: [pad.axes[0] ?? 0, -(pad.axes[1] ?? 0), pad.axes[2] ?? 0, -(pad.axes[3] ?? 0), pad.buttons[6]?.value ?? 0, pad.buttons[7]?.value ?? 0],
      });
    }
  }

  /// What the play said, to the page's console and to whoever listens.
  say() {
    const said = this.orla.text(this.x.orla_play_said(this.session));
    for (const line of said.split("\n").filter(Boolean)) {
      this.said.push(line);
      this.onsaid(line);
    }
  }

  /// One frame: the play's, shown on the page.
  async step() {
    this.pollPads();
    // With sound, the card sets the pace: a block is made when it has
    // room, as the Player's ring does.
    if (this.audio && this.block && this.held > this.block * 3) return;
    const rate = this.audio ? this.audio.context.sampleRate : 0;
    const scale = window.devicePixelRatio || 1;
    const framed = await this.frame(this.session, this.canvas.width, this.canvas.height, scale, rate);
    this.say();
    if (framed === 0) {
      await this.show();
      await this.play();
    } else if (framed < 0) {
      document.title = `Orla: ${this.orla.text(this.x.orla_error(this.session))}`;
    }
  }

  /// The picture: from the device where it can, else drawn on the CPU.
  async show() {
    const presented = await this.orla.withText(this.shown.id, (id) => this.present(this.session, id));
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

  /// The frame's sound, to the card once it is open.
  async play() {
    if (!this.audio || !this.audio.node) return;
    const field = this.x.orla_alloc(36);
    try {
      if ((await this.sound(this.session, field)) !== 0) return;
      const at = new Uint32Array(this.x.memory.buffer, field, 9);
      const [words, , , , count, stride] = at;
      const samples = new Float32Array(new Float32Array(this.x.memory.buffer, words, count * stride));
      this.block = count;
      this.audio.node.port.postMessage({ samples, channels: stride }, [samples.buffer]);
    } finally {
      this.x.orla_free(field, 36);
    }
  }

  /// Opens the sound card at the person's first touch, as a page may.
  async startSound() {
    if (this.audio) return;
    const context = new AudioContext();
    this.audio = { context, node: null };
    const url = URL.createObjectURL(new Blob([WORKLET], { type: "text/javascript" }));
    await context.audioWorklet.addModule(url);
    const node = new AudioWorkletNode(context, "orla-sound", { outputChannelCount: [2] });
    node.port.onmessage = (event) => (this.held = event.data);
    node.connect(context.destination);
    this.audio.node = node;
    this.orla.withText(`${context.sampleRate} Hz`, (text) => this.x.orla_play_sound_opened(this.session, text));
  }

  /// Frame after frame, a frame each time the page draws.
  async run() {
    if (this.running) return;
    this.running = true;
    document.title = "Orla Player";
    while (this.running) {
      try {
        await this.step();
      } catch (error) {
        this.onsaid(`stopped: ${error.message}`);
        this.running = false;
        return;
      }
      await new Promise((done) => requestAnimationFrame(done));
    }
  }

  /// Escape: the page cannot close itself, so the scene stops; a key or
  /// a click goes on.
  stop() {
    this.running = false;
    document.title = "Orla Player (stopped: a key or a click goes on)";
  }

  /// Watches the served files under `roots` and hands in the ones that
  /// changed, as a window's session watches its disk; the next frame
  /// reads them. Only a page served by serve.mjs can.
  watch(roots, every = 500) {
    let stamps = null;
    const look = async () => {
      try {
        const now = await (await fetch(`/stamps?roots=${roots.join(",")}`)).json();
        if (stamps) {
          for (const [path, stamp] of Object.entries(now)) {
            if (stamps[path] !== stamp) {
              this.orla.put(path, new Uint8Array(await (await fetch(`/${path}`)).arrayBuffer()));
            }
          }
        }
        stamps = now;
      } catch (error) {
        return;
      }
      setTimeout(look, every);
    };
    look();
  }
}

globalThis.OrlaPlayer = OrlaPlayer;
})();
