# Usage

Orla is a Rust workspace whose runtime crates have no external
dependencies; only the interface layer (the Player, `orla-tools`) takes
them. Build and verify:

```sh
cargo build --workspace
cargo test --workspace
```

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
names (`input atlas: Image = "atlas.ppm";`), which the Host watches; graph
parameters are overridden with `--param <name>=<literal>`. `--save
<node.port>=<file>` writes any node's output after the evaluation: `.field`
for any Field, `.png` or `.ppm` for a grid of pixels (a `Grid<Float>` as
gray), `.pgm` for a `Grid<Float>`, `.wav` at 48 kHz for rows of one or two
floats; the node is pinned so fusion keeps it. A Kernel that draws a
sprite sheet or synthesizes a drum, and a `--save`, are how an asset is
made; `--out` is written only when the output is a picture.

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm
```

## Inspecting a graph

```sh
cargo run -- graph examples/grayscale/scene.osn
```

prints the flattened graph in canonical form, the published Kernels with
their generations, and the plan: which groups execute, in order, with the
producers each one absorbed. In the Player, Tab cycles the preview through
the nodes (pinning each so fusion keeps it materialized) and back to the
output; the title shows the node in view, the frame time, what ran and what
was fused, and how the frame reached the window.

## Fusion

Chains of pointwise nodes run as one Kernel: a producer whose only reader
samples it at the current coordinate over the same domain is inlined into
that reader, and its image is never written. `run` reports fused groups as
`consumer+producer`. Pass `--no-fusion` to `run` to compare, or pin a node
through the protocol to observe its image.

## Backends

`run` and the Player take `--backend cpu|gpu`. The CPU interpreter is the
reference and reports per-pixel integer failures; the GPU backend runs the
same IR as WGSL compute through wgpu, keeps images resident on the device
between nodes and frames, and reads pixels back only when something
observes them. It is the one to use for anything interactive. The Player
defaults to the GPU when a device is available.

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

`--scale <n>` sets the initial window magnification, `--backend` picks the
backend, and `--input` binds external images as for `run`. On the GPU the
window is drawn from the resident output directly; the title shows the
presentation path and how many frames had to be downloaded. The pointer and
keys reach Kernels that declare `input.read`; F5 resets Graph state; Escape
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
The title reports the device's rate, the block size, and how many
callbacks ran dry.

A saved edit to a running Kernel compiles on a worker thread: the terminal
shows `preparing`, the window keeps drawing the previous version, and the
swap happens when the replacement is ready, so a republish never stalls a
frame. The same holds for fused groups, which run as separate nodes until
their fused version arrives.

## Import a model or a sound

`orla-tools` is the interface layer's converter: it speaks the Host's file
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
cargo run -p orla-tools -- import examples/model/source/crate.glb --out examples/model/assets
cargo run -p orla-tools -- import song.ogg --out assets   # writes assets/song.wav
```

## Replay tests

An application's tests are `.replay` files next to its scene: scripted frames,
pointer and key input, state resets, and expectations. No Rust is needed to
test a scene. An expectation is a pixel of the output
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

## Creation protocol

Agents drive a session over stdin and stdout with JSON lines:

```sh
cargo run -- serve examples/life/scene.osn
```

See [the protocol](PROTOCOL.md) for the operations. `orla-player --serve`
runs the same protocol inside the window, so an agent edits what a person
is watching.

## Where the contracts live

- Language syntax: [the frontend](../crates/language/src/lib.rs) and its parser tests.
- Capabilities and types: [the ABI](../crates/abi/src/lib.rs).
- IR operations and validation rules: [the IR](../crates/ir/src/lib.rs) and
  [its validator](../crates/ir/src/validate.rs).
- Document format: [the contract](DOCUMENTS.md), [the frontend](../crates/language/src/lib.rs), and [graph lowering](../crates/graph/src/document.rs).
- Publication semantics: [the runtime](../crates/runtime/src/lib.rs).
