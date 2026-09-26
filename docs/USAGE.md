# Usage

**Answers:** which command does what, with a line you can run.

`orla` prints every command and flag with no arguments; this page is the
commands with a line to run and what their flags mean. `orla player`,
`orla import` and `orla export` run `orla-player`, `orla-import` and
`orla-export` beside `orla` with the same arguments.

In a release, `orla` is `bin/orla`, and the paths below are relative to
the unpacked directory. From the source, `cargo build --release` puts
every program in `target/release/`, and `cargo run --release --` runs
`orla`:

```sh
cargo build --workspace
cargo test --workspace
```

## Start here

```sh
orla new hello
orla player hello/scene.osn
```

`orla new <dir>` writes a directory that runs: a Kernel, the scene that
places it, a replay that checks what it draws, and a README saying what
to change first. `--kind` picks the start: `picture` (the default, one
Kernel over the window), `sound` (a voice with a drawing of it) or
`game` (a grid of state the arrow keys move). Edit the Kernel while the
Player is open; a Kernel that does not compile leaves the running one
alone and says why.

## What a Kernel may call and write

```sh
orla help              # every builtin and form, one line each
orla help smoothstep   # a builtin's signatures, meaning, capability
orla help each         # a form's shape and meaning
```

The listing comes from the compiler's own tables, so it is what
compiles. The protocol's `help` answers the same as JSON; the language is
in [DOCUMENTS](DOCUMENTS.md).

## Check a Kernel

```sh
orla check --ir examples/grayscale/kernels/grayscale.orla
```

Compiles `.orla` or `.osn` files and prints their manifests, or
diagnostics at `file:line:column`. `--ir` prints the validated IR as a
backend runs it, with where work is counted (`work spend`) and looked at
(`work look`).

## Run a scene

```sh
orla run examples/grayscale/scene.osn --out output.ppm
```

Evaluates a `.osn` scene once. Flags:

- `--library <dir|file>` adds Kernel sources beyond the scene's `use`.
- `--input <name>=<file>` binds an external Field the scene declares with
  `input`; an input may instead name its own file, `bytes`, or `frames`
  (see [DOCUMENTS](DOCUMENTS.md)).
- `--param <name>=<literal>` overrides a graph parameter the scene
  declares with `param`; naming a node's own value is refused with why.
- `--window <w>x<h>[@<scale>]` is the window a scene sized by
  `domain = window();` fills, in pixels, with its scale factor.
- `--save <node.port>=<file>` writes a node's output after the
  evaluation: `.field` for any Field, `.png` or `.ppm` for pixels (a
  `Grid<Float>` as gray), `.pgm` for a `Grid<Float>`, `.wav` at 48 kHz
  for rows of one or two floats. The node is pinned so fusion keeps it.
  The name is a node, a node of a placed graph
  (`steps/8.spread/6.result`: the port follows the last dot), or a scene
  output, which stands for what it is wired to. This is how an asset is
  made.
- `--out <file>` writes the scene's picture, in the format its extension
  names. `run` writes the files named and no others.
- `--profile` prints each executed node's time, the backend that ran it
  (and why, when a GPU session sent it to the JIT), and the bytes it
  carried between Host and device. It waits for the device after each
  node, so a GPU line is the work, by the device's timestamps, or the
  work plus one round trip where the driver writes none; the lines add up
  to more than a frame.
- `--no-fusion` runs every node on its own.
- `--memory <MB>` is the Fields the Host may hold at once (2048 by
  default); `--work <M>` the million loop iterations one run of a Kernel
  may take (4000 by default). `render`, `replay`, `serve`, `graph`,
  `trace` and `bench` take both.
- `--allow file.write=<dir>` lets outputs written `to` a file
  (`output kept <- tally.result to "saves/tally.field";`) land in that
  directory; without it the session says what was not written.
  `--allow file.read=<dir>` lets a served scene's inputs read there.
  `render`, `serve` and the Player take both; a server takes its
  allowances from its command line, never from a connection. `bench` and
  `replay` write nothing.
- `--grant <capability,...>` names the capabilities a Kernel may be
  given, and no others; a Kernel requiring another is refused when
  prepared. `render`, `serve`, `bench` and the Player take it.
- `--watch` keeps running and republishes Kernels as their files change
  (see [Hot reload](#hot-reload)).
- `--backend cpu|gpu|jit` (see [Backends](#backends)).

`run` also reports fused groups as `consumer+producer`, and which `async`
nodes were still running and which landed.

```sh
orla player examples/save/scene.osn --allow file.write=examples/save/saves
```

## Render frames

```sh
orla render examples/audio/scene.osn --frames 300 --out /tmp/audio
ffmpeg -framerate 60 -i /tmp/audio/%04d.png -i /tmp/audio/sound.wav -pix_fmt yuv420p /tmp/audio.mp4
```

`orla render <scene.osn> --frames <n>` steps the scene at `--rate` (60
by default) and writes each frame's picture as `<dir>/0000.png`, ... and
the sound, block after block, as `<dir>/sound.wav` at `--sound-rate`
(48000 by default, what Kernels read as `rate()`); `--out <dir>` is
`frames` by default. The picture and sound are the outputs the Player
would choose. The clock is the frame count over the rate and `async`
runs are waited for after every frame, so a render is the same every
time. The other flags are `run`'s; `--save` writes after the last frame,
which keeps a state trained or settled over many frames as a file a
scene reads back (`input weights: Rows<Float> = "weights.field";`).

## Trace one element

```sh
orla trace examples/thumb/scene.osn --node machine --element 9352 --window 128x128
orla trace examples/life/scene.osn --node life --at 3 4
```

Evaluates the scene once (at `--time`, zero by default, under `--param`
and `--window`), then runs one element of the node's Kernel on the
interpreter: `--at x y [z]` of a map, or step `--element i` of a scan,
with the steps before it run first. It prints every `let` of the entry
in source order with its type and value, then the result, then each call
the element made (the first 256) with its own `let`s. A `let` in a branch
not taken reads `- (not run)`; a `let` in a loop shows its last
iteration. The Kernel is compiled without the optimizer, so every call
and `let` is present. The trace is of the evaluation after the one just
run, and commits nothing. The protocol's `trace` is the same as JSON.

## Packages and versions

```sh
orla package packages/text.orlapkg
```

`orla package <dir.orlapkg>` checks a package's manifest and says what
it holds; `orla pack <dir.orlapkg> [<file>]` writes it as one `.orlapkg`
file; `orla unpack <file> <new dir>` reads it back. A scene uses a
package with `use "name.orlapkg";`, by path or by name through
`ORLA_PACKAGES`. `orla version` prints `orla.api`, `orla.language`,
`orla.ir` and `orla.package`; a package needing a newer language or IR
is refused with both numbers.

## Inspect a graph

```sh
orla graph examples/grayscale/scene.osn
```

Prints the flattened graph in canonical form, the published Kernels with
their generations, and the plan: the groups that execute, in order, with
the producers each absorbed. Fusion inlines a pointwise producer into its
only reader over the same domain; `--no-fusion` turns it off.

## Measure a scene

```sh
orla bench examples/collapse/scene.osn \
    --window 640x400 --backend all --frames 20 --repeat 3 --nodes
```

Steps the scene `--frames` frames `--repeat` times on each backend and
prints the least run, in milliseconds a frame, with the median beside it
and the ratio to the first backend. The first frame of all compiles and
allocates and is discarded; every frame waits for the device; the clock
advances by frame count over `--rate`. `--nodes` measures each node too,
waiting after each, so a node placed several times is one line
(`steps/*.spread/* (48)`) and the lines add up to more than the frame.
`--json <file>` writes the measurement; `--compare <file>` reads one back
and prints the earlier number and the change. The protocol has the same
measurement ([PROTOCOL](PROTOCOL.md)).

## Backends

```sh
orla run examples/raymarch/scene.osn --backend gpu --out march.ppm
```

`run`, `render`, `replay`, `serve`, `trace` and the Player take
`--backend cpu|gpu|jit|best`; `best` is the GPU where there is a device,
else the JIT, else the interpreter, and `bench` also takes `all`. The JIT
is the default; the Player's default is `best`.

- `cpu` is the interpreter, the reference: it reports per-pixel integer
  failures and is what `trace` walks.
- `jit` compiles the IR to native code through Cranelift and runs it on
  the CPU's threads, bit for bit the interpreter's results.
- `gpu` runs the IR as WGSL compute through wgpu, keeps Fields resident
  on the device, and reads back only what is observed. A GPU session
  hands scans, and what the device cannot take, to the JIT; `run`,
  `serve` and the Player say which nodes leave and why.

## Hot reload

```sh
orla run examples/grayscale/scene.osn --out output.ppm --watch
```

Save the scene or any document it uses: the Kernels whose text changed
republish, the nodes they affect re-execute, and the output is rewritten.
A Kernel that does not compile is reported with its position and the
previous version keeps running; a Kernel removed from its document
retires.

## Player

```sh
orla player examples/material-ball/scene.osn
```

Opens a scene in a window and keeps it live, watching files as `--watch`
does; a changed Kernel compiles on a worker thread and swaps in when
ready, so a republish never stalls a frame. It takes `run`'s
`--library`, `--input`, `--backend`, `--memory`, `--work`, `--allow` and
`--grant`, and:

- `--scale <n>` sets the initial magnification; `--window <w>x<h>` the
  initial logical size.
- `--no-audio` leaves the sound card closed.
- `--replay <file>` plays a replay in the window (see
  [Replay tests](#replay-tests)).
- `--serve`, `--listen <host:port>` and `--allow-origin <origin>` serve
  the protocol from the window (see [Creation protocol](#creation-protocol)).

The window shows the first output that is a grid of pixels; the sound
card plays the first that is rows of one or two floats, one block a
frame, and paces evaluation while it plays. The window's size and scale
reach Kernels declaring `input.read` as `window()` and `scale()`, with
the pointer and keys. A gamepad is keys: the D-pad and left stick are the
arrows (37 to 40), South is Z, East is X, Start is Enter, Select is
Escape.

Keys: Tab cycles the preview through the nodes and back to the output;
F5 resets Graph state; F10 shows a line of frame data (frame interval
and rate, evaluate and present times, backend, window and scale, what
ran, was reused and was fused, `async` runs, presentation path and
downloads, the sound device's rate, block size and dry callbacks);
Escape closes the window.

## Replay tests

```sh
orla replay examples/life/tests/blinker.replay
orla player --replay examples/voxel/tests/voxel.replay
```

An application's tests are `.replay` files next to its scene, one step a
line, `#` to the end of a line a comment. Every form:

```text
scene ../scene.osn        # required; paths are relative to the replay file
library ../kernels/       # optional Kernels beyond the scene's `use`
frame 0.0                 # evaluate at this session time
settle                    # wait for every `async` run to land
pointer 20 20 down        # move the pointer and press; `up` releases
pointer up                # press or release without moving
button right down         # left, right, middle, or a pad's south, east,
button south up           # west, north, l1, r1, l3, r3, start, select
stick left 0.5 -1         # a pad's left, right or triggers pair
wheel 0 -1                # the scroll for the next frame only
window 240 135            # the window in pixels, for `window()`,
window 240 135 0.25       # and its scale factor (one when not given)
key 67 down               # a key by code; `up` releases
reset                     # forget every node's committed state
reset step.result         # forget one port's
param exposure 1.5        # set a graph parameter to a literal
expect 11 10 = 1.0 1.0 1.0        # the picture's r g b at x y, tolerance 0.01
expect 11 10 = 1.0 1.0 1.0 0.05   # with its own tolerance
expect step.result[0].score = 1   # a field of an element of a node's output
expect fall.result[0].speed > 2.5 # every component past a bound: < <= > >=
expect count ground.result = 6889 # how many elements a node's output holds
expect placed ring main           # where a node runs: main, or ordered (a
expect placed filter ordered      # scan off a GPU session's main backend)
```

A pixel is the colour the Kernel wrote, each channel a float from 0 to 1
(not 0 to 255), at `x y` from the picture's top left. Ints and bools
compare exactly, floats and vectors within the tolerance. A session with
one backend reads `ordered` as `main`. Nodes an expectation names are
pinned before the first frame, so fusion keeps them; a frame evaluates
only what its expectations need.

Key codes
follow the browser's: letters and digits by ASCII uppercase, space 32,
enter 13, arrows 37 to 40. In the Player, a replay drives the scene at
its own times, prints what it observed at the end, and leaves the window
to the person.

## Import and export

```sh
orla import examples/model/source/crate.glb --out examples/model/assets
orla import examples/skin/source/arm.glb --out examples/skin/assets
orla import song.ogg --out assets   # writes assets/song.wav
orla import examples/score/song.mid --out examples/score/assets
```

`orla-import` writes what a scene takes as inputs, into `--out <dir>`
(the source's directory by default):

- glTF or GLB: vertices as rows of `{position, normal, uv}` with node
  transforms baked, triangles as rows of three indices, the base colour
  texture as PNG; sparse accessors and JPEG textures are reported and
  skipped. A skin adds `<stem>.weights.field`, `<stem>.skeleton.field`
  and one `<stem>.<name>.clip.field` per animation, sampled at `--rate`
  frames a second (30 by default), per [the rig contract](RIGS.md),
  which the [`skin` package](../packages/skin.orlapkg) poses.
- MP3, OGG Vorbis, FLAC, AAC in MP4, Matroska audio: a WAV at the file's
  rate, one or two channels.
- Standard MIDI: `<stem>.notes.field`, a row a note (start and length in
  seconds, key, velocity 0 to 1, channel).

```sh
orla export crate.vertices.field crate.triangles.field crate.albedo.png --out crate.glb
orla export kick.field --out kick.wav --rate 48000
orla export atlas.field --out atlas.png
orla export song.notes.field --out song.mid --beats 100
```

`orla-export` writes Host files as other software's formats: vertex,
triangle and PNG as a GLB (with a rig's rows, a skin and its animations,
`--rate` 30 by default); rows of one or two floats as a WAV (`--rate` 48000
by default); a grid of pixels or floats as a PNG; rows of notes as MIDI
(`--beats` a minute, 120 by default). What `orla-import` read is written
back bit for bit.

## Creation protocol

```sh
orla serve examples/life/scene.osn
```

Agents drive a session with JSON lines on stdin and stdout;
`--listen <host:port>` also serves every connection there, and a
connection that opens with HTTP is upgraded to a WebSocket. A page
connects from this machine or a file, or from an origin
`--allow-origin` names. `--window` is the window evaluations see unless
they name one. `orla-player --serve` serves the same protocol from the
window a person is watching. The operations are in
[PROTOCOL](PROTOCOL.md).

## In your own process

```sh
cargo build --release -p orla-ffi
cc bridges/c/hello.c -Icrates/interface/ffi/include -Ltarget/release -lorla_ffi -o hello
DYLD_LIBRARY_PATH=target/release ./hello examples/text/scene.osn jit
```

`orla-ffi` builds `liborla_ffi`, dynamic and static, behind the C header
[`orla.h`](../crates/interface/ffi/include/orla.h) (`include/orla.h` and
`lib/liborla_ffi.dylib` in a release). A host opens a scene
from a path or from files it hands in, evaluates it under its own input,
and reads an output or pinned node in place as a pointer with extent,
stride, shape and schema; `orla_request` takes one protocol line. A
returned pointer is valid until the next call, a Field's words until the
next evaluate, request or close; one thread at a time per session.
[`bridges/c`](../bridges/c/README.md) is the example.

## Bridges

`bridges/blender`, `bridges/web` and `bridges/godot` are clients in
their program's language that speak the protocol over `--listen`:

```sh
orla player --serve --listen 127.0.0.1:7788 examples/mesh/scene.osn
cargo test -p orla --test bridges   # web through node and Chrome, Godot through godot -s
```

The bridge test skips, and says so, what has no tool installed.

## WebAssembly

```sh
rustup target add wasm32-wasip1
cargo build --release -p orla --target wasm32-wasip1
wasmtime run --dir . target/wasm32-wasip1/release/orla.wasm replay examples/ant/tests/ant.replay --backend cpu
```

Under WASI, `orla` runs scenes on the interpreter and `serve` answers on
stdin and stdout; there is no JIT, and an `async` run lands on the frame
that started it.

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version <the wasm-bindgen Cargo.lock holds>
cargo build --release -p orla-ffi --target wasm32-unknown-unknown
wasm-bindgen --target web --keep-lld-exports --no-typescript --out-dir target/page \
  target/wasm32-unknown-unknown/release/orla_ffi.wasm
node bridges/web/serve.mjs     # then http://127.0.0.1:7789/bridges/web/index.html
```

In a page, Orla is the C ABI with `wasm-bindgen`'s glue. Its documents
are files the page hands in (`orla_files`, `orla_files_put`,
`orla_open_files`) and it answers the protocol through `orla_request`.
It runs on its JIT, which writes WebAssembly the page loads
(`orla.load`), or on the browser's WebGPU, waiting for the device by
suspending (`orla.wait`, JavaScript Promise Integration: Chrome and Edge
137 and later) and drawing on a canvas with `orla_present`. It runs on
the page's one thread with no clock, so nothing is timed. `play.html`
is the Player in a page:

```text
http://127.0.0.1:7789/bridges/web/play.html?scene=examples/voxel/scene.osn
http://127.0.0.1:7789/bridges/web/play.html?replay=examples/voxel/tests/voxel.replay&on=gpu
```

## Where the contracts live

- Documents: [DOCUMENTS](DOCUMENTS.md), [the frontend](../crates/frontend/language/src/lib.rs), [scene lowering](../crates/frontend/scene/src/lib.rs).
- Capabilities and types: [the ABI](../crates/core/abi/src/lib.rs).
- IR and validation: [the IR](../crates/core/ir/src/lib.rs), [its validator](../crates/core/ir/src/validate.rs).
- Publication: [the Kernel Runtime](../crates/core/kernel/src/lib.rs).
