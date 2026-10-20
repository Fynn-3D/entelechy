# Entelechy bridge for the browser

A web page that is a client of Entelechy's creation protocol and nothing more.
It proves what a bridge needs: a socket that speaks the protocol, a way to
show the frame `read` returns, and a way to send what the person does.
Nothing in the Rust workspace depends on it, and it knows nothing about
how Entelechy is built.

```sh
entelechy player examples/voxel/scene.esn --serve --listen 127.0.0.1:7788 --allow-origin null
open bridges/web/index.html      # or any browser, straight from the file
```

Browsers speak WebSocket, so the listener answers a connection that opens
with an HTTP upgrade in WebSocket frames and every other connection in
plain lines; the JSON inside is the same. `entelechy serve` without a window
works the same way (`entelechy serve examples/text/scene.esn --listen
127.0.0.1:7788`); then the page's canvas is the only window, and the
session sizes window-shaped scenes by it.

- **Connect** opens the socket and asks for `status`, the Kernels, and
  the graph's params. **Run** evaluates frame after frame: every
  `evaluate` carries the pointer in output pixels, the buttons held,
  what the wheel scrolled, the keys down (the codes a Kernel reads:
  letters by uppercase ASCII, arrows 37 to 40), the canvas's size in
  device pixels as the window, and the device's scale factor; every
  frame is then `read` as a PPM and drawn, letterboxed like the Player.
  Click the picture to give it the keys. The status line shows the round
  trip and what ran.
- **Kernel** lists what the session has published; **Load** fetches a
  Kernel's source into the editor, **Prepare** compiles it and reports a
  diagnostic with its line and column, **Publish** publishes it over
  whatever generation runs; the source a session returns is what
  `publish` takes, a node's inline Kernel included. Load `scene.back@1`
  from the text example, change the color, publish, and the picture
  changes; the title itself is text in the document (`input title:
  Rows<Int> = text "ENTELECHY";`), changed through `compose`.
- **Graph params** set a parameter with `param`; a param that sizes the
  graph resizes the running scene.
- **Reset state** is the Player's F5.

The page also runs Entelechy itself, with no session to connect to: Entelechy
built as WebAssembly, and its glue from `wasm-bindgen`, in `target/page`
(docs/USAGE.md, *WebAssembly*), its documents the files the page hands
in. A page that loads modules is served, not opened from a file:
`node bridges/web/serve.mjs` serves the repository, read only, to this
machine. **Add folder** takes a folder's files by the paths they have
under it, so adding `examples` and then `packages` keeps the
repository's `../../packages/...`; the list beside it says what runs
the Kernels, `jit` (Entelechy's JIT, which writes each Kernel as WebAssembly
the page loads), `gpu` (the browser's WebGPU) or `cpu`; **Open here**
opens the scene picked, and everything above works on it, the same
protocol answered in the page. On the GPU, each frame is drawn from the
device on a second canvas over the first (`present` in `Driver`,
`entelechy_present` in Entelechy), and nothing is read back; the first canvas
still takes the pointer and the keys. What Entelechy imports from the page is
`entelechy-host.mjs` (`host` in `entelechy.js`), which the page's import map
names `entelechy`: loading a Kernel's code, and waiting for the device, which
suspends Entelechy while the page takes a turn (JavaScript Promise
Integration; Chrome and Edge 137 and later). `wasi.mjs` runs Entelechy built
for WASI in node the same way, as the runner of its WebAssembly tests.

`play.html` is the Player in a page: the picture filling the page and
nothing else, as the Player's window is, playing what its address names
(`play.html?scene=examples/voxel/scene.esn`, or
`?replay=examples/ant/tests/ant.replay`, with `on=gpu|jit|cpu`; the GPU,
else the JIT, else the interpreter when none is named). It is the page's
devices for the play every interface shares, `player.js`: keys, pointer
and wheel, the Gamepad API's pads, WebAudio (opened at the first touch,
as a page may), the canvas, and the served files it watches and hands in
again when they change. What F5, F6 and F10 mean, how a pad is a
keyboard, what the line of frame data says and how a replay drives the
scene are the play's (`entelechy_play_*`), not the page's; Tab, Escape and the
other keys are the scene's. An
agent in the page creates through `player.call(op, fields)`, the
creation protocol answered between frames as the Player's window
answers `--serve`: `write` a volume into the voxel editor's `world` and
the next frame carries on from it.

A page loads a scene's Kernels in turns: the session prepares a few each
sync and the frames wait until the scene's are prepared, so loading never
holds the page, or the device process every tab shares, in one long task.
`EntelechyPlayer.openInWorker` goes further and runs Entelechy in a worker
(`player-worker.js`): loading, preparing and every frame happen off the
page's thread, the device draws on the canvas the page hands over (an
OffscreenCanvas, which Entelechy's present finds by its id in the worker's
`entelechy_canvases`), and the page keeps only its events, the pads and the
sound card, handed to the worker as messages, with `onpreparing` hearing
how many Kernels still wait. Both players show the Kernels preparing
while they do, as the Player's window does: `entelechy_play_loading`
draws them, a seed each on a line across black. Its `call(op, fields)` is the same protocol,
answered in the worker between frames. With `trace: true` the console is
told each pipeline the browser is asked to compile, its size before and
its time after: when a device's own compiler takes the browser's device
process down, the last one told without its answer is the one.

`entelechy.js` is the client on its own: `EntelechyClient` (a request, its answer;
the session answers in order, so answers are matched first in, first
out), `decodePpm` and `decodeWords` (a frame, a Field's words), and
`Driver` (a canvas as the window and the input), and `EntelechyInPage`, the
same `call` answered by Entelechy in the page. It is a classic script,
not a module, because browsers refuse module imports between file URLs.

`test.html` checks the bridge against two serving sessions and Entelechy in
the page, and `test.mjs` runs it headless: it serves the text and voxel
examples, serves the repository's files to the page, opens the page in a
headless Chrome over the DevTools protocol, and prints the page's log,
which ends with `RESULT: ok`. It reads the text example's picture and
republishes its background, over the socket and in the page alike; in
the page it composes the scene and hands a changed one in, runs the
same checks on the page's WebGPU, and checks what the GPU presents on a
canvas; then it plays a replay in `play.html` on the page's GPU and on
its JIT to its end, and presses F5 and F6 there; then it
drives the voxel editor with a window and a pointer, places a voxel,
and resizes the window; then it opens `index.html`, connects it, loads a
Kernel into its editor, and opens the text example in it in the page;
in the page, the voxel example runs a frame on the JIT.
Node 22 or later and a Chrome, nothing installed:

```sh
cargo build --release
cargo build --release -p entelechy-ffi --target wasm32-unknown-unknown
wasm-bindgen --target web --keep-lld-exports --no-typescript --out-dir target/page \
  target/wasm32-unknown-unknown/release/entelechy_ffi.wasm
node bridges/web/test.mjs
```
