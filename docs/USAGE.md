# Usage

Orla is a Rust workspace whose runtime crates have no external
dependencies; only the interface layer (the Player, `orla-import`) takes
them, and `orla-export` needs none. Build and verify:

```sh
cargo build --workspace
cargo test --workspace
```

## What a Kernel may call

```sh
cargo run -- help              # every builtin, one line each
cargo run -- help smoothstep   # its signatures, what it does, what it needs
```

The list comes from the compiler's own tables, so it is what compiles: the
math intrinsics with every accepted signature, the Field reads, `put`,
the context functions with the capability each needs, and the vector
constructors. Over the protocol, `help` answers the same as JSON; the
language itself is in [DOCUMENTS](DOCUMENTS.md).

## Check a Kernel

Compile `.orla` files and print their manifests. Diagnostics carry
`file:line:column`. Add `--ir` to print the validated Orla IR.

```sh
cargo run -- check --ir examples/grayscale/kernels/grayscale.orla
```

## Run a scene

Evaluate a `.osn` graph and write the output image as a binary PPM. The
document names its Kernel sources with `use`; `--library <dir|file>` adds
more. External Fields declared with `input <name>: Image;` in the scene are
bound with `--input <name>=<file.ppm>`, or read from the file the input
names (`input atlas: Image = "atlas.ppm";`), which the Host watches, or
read frame by frame from a directory (`input clip: Image = frames
"clip/" at 30;`, the frame of the moment at each evaluation); graph
parameters are overridden with `--param <name>=<literal>`; a scene sized
by the window (`domain = window();`) takes `--window <w>x<h>` in pixels,
with `@<scale>` for a scale factor other than one. `--save
<node.port>=<file>` writes any node's output after the evaluation: `.field`
for any Field, `.png` or `.ppm` for a grid of pixels (a `Grid<Float>` as
gray), `.pgm` for a `Grid<Float>`, `.wav` at 48 kHz for rows of one or two
floats; the node is pinned so fusion keeps it. A Kernel that draws a
sprite sheet or synthesizes a drum, and a `--save`, are how an asset is
made; `--out` is written only when the output is a picture. `--profile`
prints how long each executed node's Kernel took. On the GPU it asks
the device for the time between the first pass of a node and the last,
by the device's own clock, and waits for the work before going on; so
the numbers are the work rather than the submission, and the run is
slower than it would be. Where a driver takes the timestamp queries and
writes nothing (Metal through wgpu, at the time of writing), the number
is the work plus one round trip to the device instead. The Player's F10
line measures the frame, which costs nothing and is the honest total.

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm
```

## Render frames

`orla render <scene.osn> --frames <n>` steps the scene at a fixed rate
(`--rate`, 60 by default) and writes each frame's picture as
`<dir>/0000.png`, `0001.png`, ... (`--out`, `frames` by default) and
the sound output, block after block, as `<dir>/sound.wav` at
`--sound-rate` (48000 by default, what Kernels read as `rate()`). The
picture is the first output that is a grid of pixels and the sound the
first that is rows of one or two floats, as the Player chooses them.
The clock is the frame count over the rate, so a render is the same
every time, and an `async` node's run is waited for after every frame,
so no step is skipped. The other flags are `run`'s; `--save
<node.port>=<file>` writes a node's output after the last frame, so a
state that took many frames to reach, a network's weights trained over
a run or a simulation settled, is kept as a file a scene reads back
with `input weights: Rows<Float> = "weights.field";`. A clip comes in
the same way, as a directory of frames a scene reads with `frames
"clip/" at <rate>`. A
video is one ffmpeg call over the directory:

```sh
cargo run --release -- render examples/audio/scene.osn --frames 300 --out /tmp/audio
ffmpeg -framerate 60 -i /tmp/audio/%04d.png -i /tmp/audio/sound.wav -pix_fmt yuv420p /tmp/audio.mp4
```

## Tracing one element

```sh
cargo run --release -- trace examples/thumb/scene.osn --node machine --element 9352 --window 128x128
cargo run -- trace examples/life/scene.osn --node life --at 3 4
```

evaluates the scene once (at `--time`, default zero, under `--param`
and `--window` as `run` takes them), then runs one element of the named
node's Kernel on the interpreter, element `--at x y [z]` of a map or step
`--element i` of a scan, the steps before it run first so the memories
and the carried value are as that step sees them, and prints every `let`
of the entry in source order with its type and value, then the result;
an `Int` of sixteen or more also in hex, a struct by its fields, an
array by its elements, a matrix by its columns, an integer vector as
`ivec2(x, y)`. A `let`
in the branch of an `if` the element did not take reads `- (not run)`; a
`let` in a loop shows its last iteration's value. Every call the element
made follows, in the order the calls returned, each with its own `let`s
and result (the first 256; a template instance is named with its Field,
`store<ram>`). What it walks is the Kernel as the document wrote it,
compiled again without the optimizer, so every call and every `let` is
there even though the run itself has the short functions written out.
The trace is of the
evaluation after the one just run (the node's `<~` inputs are the state
that evaluation committed), and commits nothing itself. Over the
protocol, `trace` is the same as JSON.

## What a Kernel may write

`orla help` lists every builtin with one line each and every form of the
language and of a graph document with its shape; `orla help <name>` is
a builtin's signatures, what it does and the capability it needs, or a
form's shape and meaning (`orla help loop`, `orla help each`, `orla help
fill`). The protocol's `help` answers the same, so an agent reads it
rather than guessing.

```sh
cargo run -- help each
```

## Packages and versions

`orla package <dir.orlapkg>` checks a package directory's manifest and
says what it holds; `orla pack <dir.orlapkg> [<file>]` writes it as one
`.orlapkg` file (Orla's own uncompressed container: a magic line, the
format version, then each file's path and bytes), and `orla unpack
<file> <new dir>` reads it back. A scene uses a package with `use
"name.orlapkg";`, by path or by name through `ORLA_PACKAGES`, and places
its graph presets by path. `orla version` prints the four versions the
plan keeps apart: `orla.api`, `orla.language`, `orla.ir`,
`orla.package`; a package's manifest names the least language and IR it
needs, and one needing more is refused with both numbers.

## Inspecting a graph

```sh
cargo run -- graph examples/grayscale/scene.osn
```

prints the flattened graph in canonical form, the published Kernels with
their generations, and the plan: which groups execute, in order, with the
producers each one absorbed. In the Player, Tab cycles the preview through
the nodes (pinning each so fusion keeps it materialized) and back to the
output; the F10 line shows the node in view, the frame time, what ran and
what was fused, and how the frame reached the window.

## Fusion

Chains of pointwise nodes run as one Kernel: a producer whose only reader
samples it at the current coordinate over the same domain is inlined into
that reader, and its image is never written. `run` reports fused groups as
`consumer+producer`. Pass `--no-fusion` to `run` to compare, or pin a node
through the protocol to observe its image.

## Backends

`run` also says which `async` nodes were still running off the frame and
which landed before the nodes ran.

`run`, `render`, `replay`, `serve`, `trace` and the Player take
`--backend cpu|gpu|jit`; the JIT is the default everywhere but the
Player, which takes the GPU when there is one and the JIT otherwise.
The CPU interpreter, `--backend cpu`, is the reference, reports
per-pixel integer failures, and is what `trace` walks;
the GPU backend runs the same IR as WGSL compute through wgpu, keeps
images resident on the device between nodes and frames, and reads pixels
back only when something observes them, and is the one to use for
anything interactive with a device; the JIT compiles the same IR to
native code through Cranelift when a Kernel is prepared and runs it on
the CPU's threads, bit for bit what the interpreter computes and ten to
thirty times faster on the examples, so a machine without a usable GPU
still runs the interactive scenes, and a machine written as a scan
steps eight to nine times faster than on the interpreter. The JIT
compiles every mode (a reduction's fold and a scatter's in-order apply
follow the interpreter's exactly); the native operations, the
rasterizer, are the CPU's code. A GPU session hands its scans to the
JIT, since a GPU walks ordered work on one thread: a machine's step
runs as native code while the picture stays on the device, the scan's
memories crossing as a few kilobytes a frame. The Player defaults to
the GPU when a device is available.

```sh
cargo run --release -- run examples/raymarch/scene.osn --backend gpu --out march.ppm
```

## Hot reload

Keep the scene running and republish Kernels as their files change:

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm --watch
```

Edit and save the scene or any document it uses. Only the Kernels whose text
changed republish and only the nodes they affect re-execute; the output file
is rewritten. A Kernel that no longer compiles is reported with its position
and the previous version keeps running. A Kernel that leaves its document
retires.

## Player

Open a scene in a window and keep it live. Kernel and scene files are watched
exactly as in `--watch`; Kernels that declare `time.read` re-render every
frame, pure nodes stay cached. Use a release build for interactive frame rates.

```sh
cargo run --release -p orla-player -- examples/material-ball/scene.osn
```

`--scale <n>` sets the initial window magnification, `--window <w>x<h>`
opens the window at that logical size instead, `--backend` picks the
backend, and `--input` binds external images as for `run`. The window's
size in physical pixels and its scale factor are context: a node with
`domain = window();` fills the window at the window's resolution and
follows a resize, and Kernels that declare `input.read` read `window()`
and `scale()`, so a toolbar laid out in units of `scale()` keeps its
size on screen. On the GPU the
window is drawn from the resident output directly; the F10 line shows the
presentation path and how many frames had to be downloaded. The pointer and
keys reach Kernels that declare `input.read`; F5 resets Graph state; F10
shows one line of frame data over the picture (the interval between
frames and the frame rate, the evaluate and present times, each
smoothed, present on the device including the wait for the display;
the backend, the window and its scale, the shown image's size, what ran,
was reused and was fused, `async` runs going, how the frame reached the
window, downloads, the sound device when open, the session time), the interval being the true frame cost since the picture is
presented on the device without a readback; Escape
closes the window.

A gamepad is another keyboard: the D-pad and the left stick are the
arrow keys (37 to 40), South (A or Cross) is Z, East (B or Circle) is X,
Start is Enter, and Select is Escape, so a scene written for keys plays
on a pad without knowing one exists. The Player reports at start whether
a pad is connected; one plugged in later is picked up.

A graph may have several outputs. The window shows the first that is a
grid of pixels; the sound card plays the first that is rows of one float
(mono) or two (stereo), one block per frame, continuing across frames as
the Kernel's `<~` state dictates. Both are claimed by structure, never by
name. The device's clock reaches Kernels as `rate()`, and while it plays
the device paces evaluation: the next block is computed when the ring can
take it, so nothing is dropped and evaluation neither races ahead of the
sound nor waits on the display. `--no-audio` leaves the sound card closed.
The F10 line reports the device's rate, the block size, and how many
callbacks ran dry.

A saved edit to a running Kernel compiles on a worker thread: the terminal
shows `preparing`, the window keeps drawing the previous version, and the
swap happens when the replacement is ready, so a republish never stalls a
frame. The same holds for fused groups, which run as separate nodes until
their fused version arrives.

## Import a model or a sound

`orla-import` is the interface layer's converter: it speaks the Host's file
formats from outside the runtime, and it is the one place besides the
Player that takes dependencies. `import` reads a glTF or GLB model and
writes what a scene takes as inputs: vertices as rows of `{position,
normal, uv}` with every node's transform baked in, triangles as rows of
three indices, and the base colour texture as a PNG, into `--out <dir>`
(the source's directory by default); skins, animations, sparse accessors
and JPEG textures are reported and left out. It reads a compressed sound
(MP3, OGG Vorbis, FLAC, AAC in MP4, Matroska audio) and writes a WAV at the
file's own rate, one or two channels, which `input` reads. The runtime
learns no format here; a model or sound in another format needs only
another tool.

```sh
cargo run -p orla-import -- examples/model/source/crate.glb --out examples/model/assets
cargo run -p orla-import -- song.ogg --out assets   # writes assets/song.wav
```

## Export a mesh, a sound, a picture

`orla-export` is the way out: it writes the Host's files as formats other
software opens. Vertex rows, triangle rows and a PNG become a GLB that
Blender and three.js load as it is; rows of one or two floats become a WAV
(`--rate` names the frames per second of a `.field`, 48000 by default; a
compressed copy is ffmpeg's job on the WAV); a grid of pixels or floats
becomes a PNG. What `orla-import` brought in leaves the same, bit for bit.

```sh
cargo run -p orla-export -- crate.vertices.field crate.triangles.field crate.albedo.png --out crate.glb
cargo run -p orla-export -- kick.field --out kick.wav --rate 48000
cargo run -p orla-export -- atlas.field --out atlas.png
```

## Replay tests

An application's tests are `.replay` files next to its scene: scripted frames,
pointer and key input, the window (`window 240 135 0.25`: its size in
pixels and its scale factor), state resets, `settle` (waits for every
`async` node's run off the frame to land, so the next frame reads it),
and expectations. No Rust is
needed to test a scene. An expectation is a pixel of the output
(`expect 11 10 = 1.0 1.0 1.0`, with an optional tolerance) or a field of an
element of a node's output, named as the Kernel declared it
(`expect step.result[0].score = 1`, `expect mix.result[0] = -0.125 -0.125 0.01`):
ints and bools compare exactly, floats and vectors within the trailing
tolerance. The second kind states what the application's state is rather
than what its picture looks like, and nodes it names are kept materialized.

```sh
cargo run -- replay examples/life/tests/blinker.replay
```

The file format is documented at the top of
[the replay module](../crates/cli/src/replay.rs). Key codes follow the common
browser codes: letters and digits by ASCII uppercase, space 32, enter 13,
arrows 37 to 40.

## Documents

[The document format](DOCUMENTS.md) defines `.orla` libraries and `.osn`
graphs: one grammar, wiring written at the consumer, inline Kernels, and
`use` for reuse.

## Bridges

A running window is open to other programs over the protocol's socket
transport: `orla-player --serve --listen 127.0.0.1:7788 scene.osn`, or
`orla serve --listen ...` without a window. `bridges/blender` is an add-on
for Blender that pushes a mesh into a scene's assets as the Host's files
(and keeps pushing while you edit), pulls a node's rows out as a mesh and
the picture as an image, sets parameters and publishes Kernels from the
text editor; `bridges/web` is the second: a page, opened from the file,
that connects over a WebSocket (the listener upgrades a connection that
opens with HTTP), shows the frames, sends the pointer, keys and the
canvas as the window, and edits and publishes Kernels; `bridges/godot`
is the third: a Godot 4 project whose `OrlaView` Control is the window
(the frames drawn as a texture, the pointer, keys and its size sent with
every evaluate), with the graph's params as controls and a Kernel's rows
pulled out as an `ArrayMesh`. A bridge is a client in its host's
language, outside the workspace, speaking only the public contracts.

## In your own process

The same session is a library: `orla-ffi` builds `liborla_ffi`
(a dynamic and a static library) behind the C header
[`crates/ffi/include/orla.h`](../crates/ffi/include/orla.h), ten
functions any language binds. A host opens a scene on a backend,
evaluates it under its own input (time, pointer, buttons, keys, the
window and its scale), and reads an output or a pinned node as a
pointer into the Host's words with its extent, stride, shape and schema:
a picture is RGBA32F in place, ready for a texture, no socket and no
encoding. `orla_request` takes one line of the protocol and returns its
answer, so params, publishing, composing and resets are there too.

```sh
cargo build --release -p orla-ffi
cc bridges/c/hello.c -Icrates/ffi/include -Ltarget/release -lorla_ffi -o hello
DYLD_LIBRARY_PATH=target/release ./hello examples/text/scene.osn jit
```

[`bridges/c`](../bridges/c/README.md) is that example. A pointer the
library returns belongs to the session until the next call; a Field's
words until the next evaluate, request or close; one thread at a time
per session.

## Creation protocol

Agents drive a session over stdin and stdout with JSON lines:

```sh
cargo run -- serve examples/life/scene.osn
```

A job that spans many frames is one `advance` request: it runs the
frames, stops when a value says to, and saves what was asked for. See
[the protocol](PROTOCOL.md) for the operations. `orla-player --serve`
runs the same protocol inside the window, so an agent edits what a person
is watching.

## Where the contracts live

- Language syntax: [the frontend](../crates/language/src/lib.rs) and its parser tests.
- Capabilities and types: [the ABI](../crates/abi/src/lib.rs).
- IR operations and validation rules: [the IR](../crates/ir/src/lib.rs) and
  [its validator](../crates/ir/src/validate.rs).
- Document format: [the contract](DOCUMENTS.md), [the frontend](../crates/language/src/lib.rs), and [graph lowering](../crates/graph/src/document.rs).
- Publication semantics: [the runtime](../crates/runtime/src/lib.rs).
