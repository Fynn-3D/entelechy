// Entelechy's Player in a page: the page's devices for the play every
// interface shares (`entelechy_session::play`, reached through the C ABI's
// `entelechy_play_*`). What the keys mean, how a pad is a keyboard, what the
// line of frame data says, which output is the picture and which the
// sound, and how a replay drives the scene are the play's; this is only
// the canvas, the page's events, the Gamepad API, WebAudio, and the
// served files watched and handed in. A classic script after entelechy.js; it
// defines `EntelechyPlayer`.

(() => {
const { EntelechyInPage } = Entelechy;

/// The Gamepad API's standard mapping, button by button, in the play's
/// names; the triggers (6, 7) are axes.
const PAD_BUTTONS = [
  "south", "east", "west", "north", "left_shoulder", "right_shoulder", null, null,
  "select", "start", "left_thumb", "right_thumb", "up", "down", "left", "right",
];

/// A page's pointer button as the play's: 0 left, 1 right, 2 middle.
const BUTTON = { 0: 0, 1: 2, 2: 1 };

/// Keys the page would act on itself, which the scene has instead.
/// The blocks the card holds at most, as the Player's ring.
const BLOCKS_AHEAD = 4;

/// `canvas` made to show the pointer `look` (`EntelechyInPage.pointerLook`):
/// the scene's picture as the page's pointer, its hotspot where it
/// points, or the page's own again.
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
/// What a step answers when the card is still ahead: no frame was made.
const AHEAD = "ahead";

/// What a key press writes, if it writes a character: its own text, when
/// no command is held with it.
function written(event) {
  return event.key.length === 1 && !event.ctrlKey && !event.metaKey ? event.key : "";
}

const HELD = new Set(["Tab", "Escape", "F5", "F6", "F10", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", " "]);

/// The sound card: a worklet that plays the blocks it is handed, in
/// order, and says how many samples it holds, which paces the frames as
/// the Player's ring does.
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

class EntelechyPlayer {
  /// Plays `entelechy` (an EntelechyInPage with its scene open) on `canvas`, the
  /// page's window, with `shown` a canvas over it that takes no pointer,
  /// where the GPU presents.
  constructor(entelechy, canvas, shown) {
    this.entelechy = entelechy;
    this.x = entelechy.x;
    this.canvas = canvas;
    this.shown = shown;
    this.context = canvas.getContext("2d");
    const promising = (name) =>
      typeof WebAssembly.promising === "function" ? WebAssembly.promising(this.x[name]) : this.x[name];
    this.frame = promising("entelechy_play_frame");
    this.present = promising("entelechy_play_present");
    this.pixels = promising("entelechy_play_pixels");
    this.sound = promising("entelechy_play_sound");
    this.running = false;
    this.pads = new Map();
    this.audio = null;
    this.held = 0;
    // The samples the card may hold, as the Player's ring: four of the
    // first block (`BLOCKS_AHEAD`), set when it comes.
    this.ring = 0;
    this.said = [];
    this.onsaid = (line) => console.log(line);
    // What the person did, handed to the play between frames: a frame
    // waiting on the device has Entelechy suspended, and nothing may reach
    // into it until it is done.
    this.hands = [];
    // Requests of the creation protocol, answered between frames as the
    // Player's window answers `--serve`: an agent in the page creates in
    // the world a person is playing.
    this.requests = [];
    this.listen();
  }

  /// Opens a scene or a replay (paths among `files`, [path, bytes]
  /// pairs, or [path, length] for a file the page's server keeps, read
  /// from there when it is read), on `backend`, or as the Player does
  /// when none is named: the GPU, else the JIT, else the interpreter.
  static async open({ glue, host, wasm, files, scene, replay, backend, canvas, shown }) {
    const entelechy = await EntelechyInPage.load(glue, host, wasm);
    globalThis.entelechy_files_base = new URL(".", location.href).href;
    for (const [path, bytes] of files) {
      if (typeof bytes === "number") entelechy.putElsewhere(path, bytes);
      else entelechy.put(path, bytes);
    }
    // "best" is Entelechy's to decide, as for the Player: the backends it
    // passed over are among what the play says.
    const on = backend || "best";
    if (replay) await entelechy.openReplay(replay, on);
    else await entelechy.open(scene, on);
    return new EntelechyPlayer(entelechy, canvas, shown);
  }

  get session() {
    return this.entelechy.session;
  }

  /// Hands characters typed to the play.
  typed(text) {
    this.entelechy.withText(text, (at) => this.x.entelechy_play_typed(this.session, at));
  }

  /// Hands a key to the play.
  key(name, down) {
    this.entelechy.withText(name, (at) => this.x.entelechy_play_key(this.session, at, down ? 1 : 0));
  }

  /// One request of the creation protocol (`write`, `publish`, `read`,
  /// ...), answered between frames, or in its own turn while the scene is
  /// stopped; the same promise `EntelechyInPage.call` gives. Everything that
  /// reaches into Entelechy, a frame, a hand, a request, waits for its turn,
  /// so nothing reaches in while a call inside waits for the device.
  call(op, fields = {}) {
    const asked = new Promise((resolve, reject) => this.requests.push({ op, fields, resolve, reject }));
    if (!this.running) this.entelechy.inTurn(() => this.answerNow());
    return asked;
  }

  /// Answers the requests that came since the last turn, inside one.
  async answerNow() {
    const requests = this.requests;
    this.requests = [];
    for (const { op, fields, resolve, reject } of requests) {
      await this.entelechy.callNow(op, fields).then(resolve, reject);
    }
  }

  /// What the person did since the last frame, handed to the play now.
  handOver() {
    const hands = this.hands;
    this.hands = [];
    for (const hand of hands) hand();
    return true;
  }

  listen() {
    window.addEventListener("keydown", (event) => {
      // A key that writes a character is typed as well as pressed, and
      // typed again each time it repeats: what a scene's `typed()` reads.
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
      // Lines, y positive away from the person.
      const [x, y] = [event.deltaX * unit, -event.deltaY * unit];
      this.hands.push(() => this.x.entelechy_play_wheel(this.session, x, y));
    }, { passive: false });
    // Both canvases are the window: sized by the one always shown, since
    // the other is hidden, and so sizeless, until the device draws on it.
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

  /// Hands the pads' buttons and axes to the play, where they changed,
  /// and how many are connected.
  pollPads() {
    let connected = 0;
    for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
      if (!pad || pad.mapping !== "standard") continue;
      connected += 1;
      const last = this.pads.get(pad.index) || { buttons: [], axes: [] };
      const axis = (name, value, at) => {
        if (last.axes[at] !== value) {
          this.entelechy.withText(name, (text) => this.x.entelechy_play_axis(this.session, text, value));
        }
      };
      pad.buttons.forEach((button, at) => {
        const name = PAD_BUTTONS[at];
        if (name && last.buttons[at] !== button.pressed) {
          this.entelechy.withText(name, (text) => this.x.entelechy_play_pad(this.session, text, button.pressed ? 1 : 0));
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
    this.x.entelechy_play_pads(this.session, connected);
  }

  /// What the play said, to the page's console and to whoever listens.
  say() {
    const said = this.entelechy.text(this.x.entelechy_play_said(this.session));
    for (const line of said.split("\n").filter(Boolean)) {
      this.said.push(line);
      this.onsaid(line);
    }
  }

  /// One frame: the play's, shown on the page, in its turn.
  step() {
    return this.entelechy.inTurn(() => this.stepNow());
  }

  async stepNow() {
    if (!this.handOver()) return;
    this.pollPads();
    await this.answerNow();
    this.say();
    // With sound, the card sets the pace, as the Player's ring does: a
    // frame is made while its ring has room for a frame's worth at 60 Hz,
    // and asked again soon while it has not.
    if (this.audio && this.ring && this.held + this.audio.context.sampleRate / 60 > this.ring) return AHEAD;
    const rate = this.audio ? this.audio.context.sampleRate : 0;
    const scale = window.devicePixelRatio || 1;
    // What the card holds, which the frame's `queued()` reads.
    this.x.entelechy_play_sound_queued(this.session, this.audio ? this.held : 0);
    const framed = await this.frame(this.session, this.canvas.width, this.canvas.height, scale, rate);
    this.say();
    if (framed === 0) {
      await this.show();
      await this.play();
      wear(this.canvas, this.entelechy.pointerLook());
    } else if (framed < 0) {
      // The title holds what the page has room for; the console the whole.
      const error = this.entelechy.text(this.x.entelechy_error(this.session));
      document.title = `Entelechy: ${error}`;
      if (error !== this.failed) console.error(`Entelechy: ${error}`);
      this.failed = error;
    }
  }

  /// The picture: from the device where it can, else drawn on the CPU.
  async show() {
    const presented = await this.entelechy.withText(this.shown.id, (id) => this.present(this.session, id));
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
    const field = this.x.entelechy_alloc(36);
    try {
      if ((await this.sound(this.session, field)) !== 0) return;
      const at = new Uint32Array(this.x.memory.buffer, field, 9);
      const [words, , , , count, stride] = at;
      const samples = new Float32Array(new Float32Array(this.x.memory.buffer, words, count * stride));
      // The block it is sized by is at least two frames' worth at 60 Hz,
      // as the Player's: a first block that came short leaves room.
      if (!this.ring) this.ring = Math.max(count, Math.floor(this.audio.context.sampleRate / 30), 64) * BLOCKS_AHEAD;
      this.audio.node.port.postMessage({ samples, channels: stride }, [samples.buffer]);
    } finally {
      this.x.entelechy_free(field, 36);
    }
  }

  /// Opens the sound card at the person's first touch, as a page may; the
  /// play hears of it between frames.
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
        this.hands.push(() => this.entelechy.withText(described, (text) => this.x.entelechy_play_sound_opened(this.session, text)));
      } catch (error) {
        await context.close().catch(() => {});
        throw error;
      } finally {
        this.startingSound = null;
      }
    })();
    return this.startingSound;
  }

  /// Frame after frame, a frame each time the page draws.
  async run() {
    if (this.running) return;
    this.running = true;
    document.title = "Entelechy Player";
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
              const bytes = new Uint8Array(await (await fetch(`/${path}`)).arrayBuffer());
              this.hands.push(() => this.entelechy.put(path, bytes));
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

/// The Player with Entelechy in a worker (player-worker.js): the page keeps
/// its events, the pads and the sound card, and hands them over; loading,
/// preparing and every frame run in the worker, so they never hold the
/// page. The device draws on `shown`, handed to the worker; a frame drawn
/// on the CPU comes back to be put on `canvas`.
class EntelechyWorkerPlayer {
  constructor(worker, canvas, shown) {
    this.worker = worker;
    this.canvas = canvas;
    this.shown = shown;
    this.context = canvas.getContext("2d");
    this.said = [];
    this.onsaid = (line) => console.log(line);
    this.onpreparing = null;
    this.audio = null;
    this.asked = new Map();
    this.next = 0;
    this.pads = "";
  }

  /// Opens a scene or a replay in a new worker, as `EntelechyPlayer.open`
  /// opens one in the page; `worker` is the URL of player-worker.js,
  /// `onpreparing` hears how many Kernels still wait to prepare; `trace`
  /// has the console told each pipeline the device is asked to compile,
  /// before and after. Resolves once the scene has loaded and its frames
  /// run.
  static openInWorker({ worker, glue, files, scene, replay, backend, canvas, shown, onpreparing, trace }) {
    const at = new URL(worker || "player-worker.js", location.href);
    const host = new URL("entelechy-host.mjs", at).href;
    const player = new EntelechyWorkerPlayer(new Worker(at), canvas, shown);
    player.onpreparing = onpreparing || null;
    const offscreen = shown.transferControlToOffscreen();
    shown.style.display = "block";
    return new Promise((resolve, reject) => {
      player.worker.onmessage = ({ data }) => player.heard(data, resolve, reject);
      player.worker.onerror = (event) => reject(new Error(event.message || "the worker failed"));
      // A file given as its length stays with the page's server; the
      // worker reads it from there when it is read.
      const buffers = files.filter(([, bytes]) => typeof bytes !== "number").map(([, bytes]) => bytes.buffer);
      player.worker.postMessage(
        {
          open: { glue: new URL(glue, location.href).href, host, files, base: new URL(".", location.href).href, scene, replay, backend, canvas: offscreen, id: shown.id, trace: !!trace },
          size: player.fit(),
        },
        [offscreen, ...buffers],
      );
      player.listen();
    });
  }

  /// The window's size in device pixels: both canvases, and the worker's.
  fit() {
    const scale = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(this.canvas.clientWidth * scale));
    const height = Math.max(1, Math.round(this.canvas.clientHeight * scale));
    this.canvas.width = width;
    this.canvas.height = height;
    return { width, height, scale };
  }

  heard(data, opened, refused) {
    if (data.said !== undefined) {
      this.said.push(data.said);
      this.onsaid(data.said);
    } else if (data.preparing !== undefined) {
      if (this.onpreparing) this.onpreparing(data.preparing);
    } else if (data.opened) {
      opened(this);
    } else if (data.presented !== undefined) {
      this.shown.style.display = data.presented ? "block" : "none";
    } else if (data.pointer !== undefined) {
      wear(this.canvas, data.pointer);
    } else if (data.pixels) {
      const image = this.context.createImageData(data.width, data.height);
      for (let i = 0; i < data.pixels.length; i++) {
        const word = data.pixels[i];
        image.data[i * 4] = (word >> 16) & 255;
        image.data[i * 4 + 1] = (word >> 8) & 255;
        image.data[i * 4 + 2] = word & 255;
        image.data[i * 4 + 3] = 255;
      }
      this.context.putImageData(image, 0, 0);
    } else if (data.failed !== undefined) {
      // The title holds what the page has room for; the console the whole.
      document.title = `Entelechy: ${data.failed}`;
      console.error(`Entelechy: ${data.failed}`);
      if (data.fatal) refused(new Error(data.failed));
    } else if (data.stopped !== undefined) {
      this.onsaid(`stopped: ${data.stopped}`);
    } else if (data.answer !== undefined) {
      const asked = this.asked.get(data.answer);
      this.asked.delete(data.answer);
      if (!asked) return;
      if (data.error) {
        const error = new Error(data.error.message);
        error.line = data.error.line;
        error.column = data.error.column;
        asked.reject(error);
      } else {
        asked.resolve(data.ok);
      }
    }
  }

  /// One request of the creation protocol, answered between frames in
  /// the worker; the same promise `EntelechyPlayer.call` gives.
  call(op, fields = {}) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.asked.set(id, { resolve, reject });
      this.worker.postMessage({ call: { id, op, fields } });
    });
  }

  /// Hands a file in again, as one changed on disk: the next sync reads it.
  put(path, bytes) {
    this.worker.postMessage({ put: path, bytes });
  }

  listen() {
    const send = (message) => this.worker.postMessage(message);
    window.addEventListener("keydown", (event) => {
      const text = written(event);
      if (text) send({ typed: text });
      if (event.repeat) return;
      if (HELD.has(event.key)) event.preventDefault();
      this.startSound().catch(() => {});
      send({ key: event.key, down: true });
    });
    window.addEventListener("keyup", (event) => {
      if (HELD.has(event.key)) event.preventDefault();
      send({ key: event.key, down: false });
    });
    const scale = () => window.devicePixelRatio || 1;
    // The page's canvas takes the pointer; the worker's draws over it.
    for (const target of [this.canvas, this.shown]) {
      target.addEventListener("pointermove", (event) => send({ pointer: [event.offsetX * scale(), event.offsetY * scale()] }));
      target.addEventListener("pointerdown", (event) => {
        this.startSound().catch(() => {});
        target.setPointerCapture(event.pointerId);
        send({ button: BUTTON[event.button] ?? 0, down: true });
      });
      target.addEventListener("pointerup", (event) => send({ button: BUTTON[event.button] ?? 0, down: false }));
      target.addEventListener("contextmenu", (event) => event.preventDefault());
      target.addEventListener("wheel", (event) => {
        event.preventDefault();
        const unit = event.deltaMode === 1 ? 1 : 1 / 40;
        send({ wheel: [event.deltaX * unit, -event.deltaY * unit] });
      }, { passive: false });
    }
    new ResizeObserver(() => send({ size: this.fit() })).observe(this.canvas);
    // The pads, read each time the page draws, handed over when they change.
    const poll = () => {
      const seen = [];
      for (const pad of navigator.getGamepads ? navigator.getGamepads() : []) {
        if (!pad || pad.mapping !== "standard") continue;
        seen.push({
          index: pad.index,
          buttons: pad.buttons.map((button) => button.pressed),
          // A page's stick reads down as positive; the play's up is.
          axes: [pad.axes[0] ?? 0, -(pad.axes[1] ?? 0), pad.axes[2] ?? 0, -(pad.axes[3] ?? 0), pad.buttons[6]?.value ?? 0, pad.buttons[7]?.value ?? 0],
        });
      }
      const now = JSON.stringify(seen);
      if (now !== this.pads) {
        this.pads = now;
        send({ pads: seen });
      }
      requestAnimationFrame(poll);
    };
    requestAnimationFrame(poll);
  }

  /// Opens the sound card at the person's first touch, as a page may, and
  /// hands its worklet's port to the worker, which plays into it.
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
        node.connect(context.destination);
        this.audio = { context, node };
        this.worker.postMessage({ sound: node.port, rate: context.sampleRate }, [node.port]);
      } catch (error) {
        await context.close().catch(() => {});
        throw error;
      } finally {
        this.startingSound = null;
      }
    })();
    return this.startingSound;
  }
}

EntelechyPlayer.openInWorker = (options) => EntelechyWorkerPlayer.openInWorker(options);

globalThis.EntelechyPlayer = EntelechyPlayer;
globalThis.EntelechyWorkerPlayer = EntelechyWorkerPlayer;
})();
