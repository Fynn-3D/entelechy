# Orla bridge for the browser

A web page that is a client of Orla's creation protocol and nothing more.
It proves what a bridge needs: a socket that speaks the protocol, a way to
show the frame `read` returns, and a way to send what the person does.
Nothing in the Rust workspace depends on it, and it knows nothing about
how Orla is built.

```sh
cargo run --release -p orla-player -- examples/voxel/scene.osn --serve --listen 127.0.0.1:7788
open bridges/web/index.html      # or any browser, straight from the file
```

Browsers speak WebSocket, so the listener answers a connection that opens
with an HTTP upgrade in WebSocket frames and every other connection in
plain lines; the JSON inside is the same. `orla serve` without a window
works the same way (`cargo run -- serve examples/text/scene.osn --listen
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
  whatever generation runs. An inline Kernel's source comes as its node
  wrote it, without the id the node gave it and perhaps with the node's
  `domain`, so Load writes the id in and leaves the domain out. Load
  `scene.title@1` from the text example, change a code, publish, and the
  label changes.
- **Graph params** set a parameter with `param`; a param that sizes the
  graph resizes the running scene.
- **Reset state** is the Player's F5.

`orla.js` is the client on its own: `OrlaClient` (a request, its answer;
the session answers in order, so answers are matched first in, first
out), `decodePpm` and `decodeWords` (a frame, a Field's words), and
`Driver` (a canvas as the window and the input). It is a classic script,
not a module, because browsers refuse module imports between file URLs.

`test.html` checks the bridge against two serving sessions, and
`test.mjs` runs it headless: it serves the text and voxel examples,
opens the page in a headless Chrome over the DevTools protocol, and
prints the page's log, which ends with `RESULT: ok`. It reads the text
example's picture and republishes its title, then drives the voxel
editor with a window and a pointer, places a voxel, and resizes the
window; then it opens `index.html`, connects it, and loads a Kernel into
its editor. Node 22 or later and a Chrome, nothing installed:

```sh
cargo build --release
node bridges/web/test.mjs
```
