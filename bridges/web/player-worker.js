// Entelechy's Player in a worker: Entelechy itself (WebAssembly), its Kernels
// prepared in turns, and the play's frames, all off the page's thread, so a
// scene loading or a frame waiting for the device never holds the page.
// The page keeps what only a page has (`EntelechyPlayer.openInWorker` in
// player.js): its events, the Gamepad API and the sound card, handed here
// as messages; the device draws on the canvas the page handed over (an
// OffscreenCanvas, kept in `entelechy_canvases` by its id, where
// Entelechy's present finds it). A classic worker script beside entelechy.js.

importScripts("entelechy.js");

const { EntelechyInPage } = Entelechy;

/// The play's names for the Gamepad API's standard buttons; the triggers
/// (6, 7) are axes.
const PAD_BUTTONS = [
  "south", "east", "west", "north", "left_shoulder", "right_shoulder", null, null,
  "select", "start", "left_thumb", "right_thumb", "up", "down", "left", "right",
];

self.entelechy_canvases = {};

let entelechy = null;
let x = null;
let frame, present, pixels, sound;
let shown = "";
let size = { width: 1, height: 1, scale: 1 };
// What the page handed over since the last frame, handed to the play then:
// a frame waiting on the device has Entelechy suspended.
let hands = [];
// Requests of the creation protocol, answered between frames.
let requests = [];
let pads = new Map();
let sounding = null;
let held = 0;
// The samples the card may hold, as the Player's ring: four of the first
// block (`BLOCKS_AHEAD`), set when it comes.
let ring = 0;
let failed = "";
let running = false;

/// The blocks the card holds at most, as the Player's ring.
const BLOCKS_AHEAD = 4;
/// What a step answers when the card is still ahead: no frame was made.
const AHEAD = "ahead";

const post = (message, transfer) => self.postMessage(message, transfer || []);

function say() {
  const said = entelechy.text(x.entelechy_play_said(entelechy.session));
  for (const line of said.split("\n").filter(Boolean)) post({ said: line });
}

function withText(text, use) {
  return entelechy.withText(text, use);
}

/// The pads as the page last saw them: what changed is handed to the play.
function padsNow(seen) {
  const session = entelechy.session;
  for (const pad of seen) {
    const last = pads.get(pad.index) || { buttons: [], axes: [] };
    pad.buttons.forEach((pressed, at) => {
      const name = PAD_BUTTONS[at];
      if (name && last.buttons[at] !== pressed) withText(name, (t) => x.entelechy_play_pad(session, t, pressed ? 1 : 0));
    });
    ["left_x", "left_y", "right_x", "right_y", "left_trigger", "right_trigger"].forEach((name, at) => {
      if (last.axes[at] !== pad.axes[at]) withText(name, (t) => x.entelechy_play_axis(session, t, pad.axes[at]));
    });
    pads.set(pad.index, pad);
  }
  x.entelechy_play_pads(session, seen.length);
}

async function answer() {
  const asked = requests;
  requests = [];
  for (const { id, op, fields } of asked) {
    try {
      post({ answer: id, ok: await entelechy.callNow(op, fields) });
    } catch (error) {
      post({ answer: id, error: { message: error.message, line: error.line, column: error.column } });
    }
  }
}

async function step() {
  const session = entelechy.session;
  for (const hand of hands.splice(0)) hand();
  await answer();
  say();
  // With sound, the card sets the pace, as the Player's ring does: a
  // frame is made while its ring has room for a frame's worth at 60 Hz,
  // and asked again soon (`AHEAD`) while it has not.
  if (sounding && ring && held + sounding.rate / 60 > ring) return AHEAD;
  // What the card holds, which the frame's `queued()` reads.
  x.entelechy_play_sound_queued(session, sounding ? held : 0);
  const framed = await frame(session, size.width, size.height, size.scale, sounding ? sounding.rate : 0);
  say();
  if (framed === 0) {
    await show();
    await play();
  } else if (framed === 1) {
    const left = x.entelechy_play_preparing(session);
    post({ preparing: left });
    if (left > 0) showLoading();
  } else if (framed < 0) {
    const error = entelechy.text(x.entelechy_error(session));
    if (error !== failed) post({ failed: error });
    failed = error;
  }
}

/// The picture: from the device on the handed-over canvas, else drawn on
/// the CPU and handed to the page.
async function show() {
  const session = entelechy.session;
  const presented = await withText(shown, (id) => present(session, id));
  post({ presented: presented === 0 });
  if (presented === 0) return;
  const pointer = await pixels(session, size.width, size.height);
  if (!pointer) return;
  const words = new Uint32Array(x.memory.buffer, pointer, size.width * size.height).slice();
  post({ pixels: words, width: size.width, height: size.height }, [words.buffer]);
}

/// The Kernels preparing, drawn on the CPU and handed to the page, at most
/// thirty times a second.
let loadingDrawn = 0;
function showLoading() {
  const now = performance.now();
  if (now - loadingDrawn < 33 || !entelechy.session) return;
  loadingDrawn = now;
  const words = entelechy.loading(size.width, size.height, size.scale);
  if (!words) return;
  post({ presented: false });
  post({ pixels: words, width: size.width, height: size.height }, [words.buffer]);
}

/// The frame's sound, to the card's worklet through the port the page
/// handed over.
async function play() {
  if (!sounding) return;
  const field = x.entelechy_alloc(36);
  try {
    if ((await sound(entelechy.session, field)) !== 0) return;
    const [words, , , , count, stride] = new Uint32Array(x.memory.buffer, field, 9);
    const samples = new Float32Array(new Float32Array(x.memory.buffer, words, count * stride));
    // The block it is sized by is at least two frames' worth at 60 Hz,
    // as the Player's: a first block that came short leaves room.
    if (!ring) ring = Math.max(count, Math.floor(sounding.rate / 30), 64) * BLOCKS_AHEAD;
    sounding.port.postMessage({ samples, channels: stride }, [samples.buffer]);
  } finally {
    x.entelechy_free(field, 36);
  }
}

const next = (work) =>
  typeof self.requestAnimationFrame === "function" ? self.requestAnimationFrame(work) : setTimeout(work, 16);

function run() {
  if (running) return;
  running = true;
  const loop = async () => {
    let stepped;
    try {
      stepped = await entelechy.inTurn(step);
    } catch (error) {
      post({ stopped: error.message });
      running = false;
      return;
    }
    if (stepped === AHEAD) setTimeout(loop, 1);
    else next(loop);
  };
  next(loop);
}

async function open({ glue, host, files, base, scene, replay, backend, canvas, id, trace }) {
  // Entelechy's device tells the console each pipeline it compiles.
  self.entelechy_trace = !!trace;
  entelechy = await EntelechyInPage.load(glue, host);
  x = entelechy.x;
  const promising = (name) =>
    typeof WebAssembly.promising === "function" ? WebAssembly.promising(x[name]) : x[name];
  frame = promising("entelechy_play_frame");
  present = promising("entelechy_play_present");
  pixels = promising("entelechy_play_pixels");
  sound = promising("entelechy_play_sound");
  shown = id;
  self.entelechy_canvases[id] = canvas;
  canvas.width = size.width;
  canvas.height = size.height;
  // A file given as its length stays with the page, read from `base`.
  self.entelechy_files_base = base;
  for (const [path, bytes] of files) {
    if (typeof bytes === "number") entelechy.putElsewhere(path, bytes);
    else entelechy.put(path, bytes);
  }
  entelechy.onpreparing = (left) => {
    post({ preparing: left });
    showLoading();
  };
  const on = backend || "best";
  if (replay) await entelechy.openReplay(replay, on);
  else await entelechy.open(scene, on);
  say();
  post({ opened: true });
  run();
}

self.onmessage = ({ data }) => {
  if (data.open) {
    size = data.size;
    open(data.open).catch((error) => post({ failed: error.message, fatal: true }));
    return;
  }
  if (data.size) {
    size = data.size;
    const canvas = self.entelechy_canvases[shown];
    if (canvas) {
      canvas.width = size.width;
      canvas.height = size.height;
    }
    return;
  }
  if (data.call) {
    requests.push(data.call);
    return;
  }
  if (!entelechy || !entelechy.session) return;
  const session = () => entelechy.session;
  if (data.key) hands.push(() => withText(data.key, (t) => x.entelechy_play_key(session(), t, data.down ? 1 : 0)));
  else if (data.pointer) hands.push(() => x.entelechy_play_pointer(session(), data.pointer[0], data.pointer[1]));
  else if (data.button !== undefined) hands.push(() => x.entelechy_play_button(session(), data.button, data.down ? 1 : 0));
  else if (data.wheel) hands.push(() => x.entelechy_play_wheel(session(), data.wheel[0], data.wheel[1]));
  else if (data.pads) hands.push(() => padsNow(data.pads));
  else if (data.put) hands.push(() => entelechy.put(data.put, data.bytes));
  else if (data.sound) {
    sounding = { port: data.sound, rate: data.rate };
    sounding.port.onmessage = (event) => (held = event.data);
    hands.push(() => withText(`${data.rate} Hz`, (t) => x.entelechy_play_sound_opened(session(), t)));
  }
};
