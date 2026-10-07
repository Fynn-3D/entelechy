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

## Syntax and API

```sh
orla help              # every layer's syntax and API, and the commands
orla help smoothstep   # one name in full: a builtin's signatures and capability
orla help window       # every item of a name: the builtin, the replay line, the operation
orla help publish --json
```

Every answer comes from the table the part it describes executes: the
compiler's builtins and forms, what each capability grants from the ABI,
the replayer's lines, the protocol's operations, the command line's
commands. The protocol's `help` answers the same, without `commands`
([REFERENCE](REFERENCE.md)); the language is in
[DOCUMENTS](DOCUMENTS.md), whole programs in `examples/`.

## Check a Kernel

```sh
orla check --ir examples/grayscale/kernels/grayscale.orla
```

Compiles `.orla` or `.osn` files and prints their manifests, or
diagnostics at `file:line:column`, every fault of a document's functions
in one run. A scene's nodes are held to their
Kernels' ports as an evaluation binds them, in the same words: each port
a node names is declared, each input bound, each parameter of its type,
each `value(port)` of its domain an input of numbers; what only the data
decides, such as how many elements `value` reads from, is the
evaluation's. `--ir` prints the validated IR as a
backend runs it, with where work is counted (`work spend`) and looked at
(`work look`). A Kernel that carries more than 5 000 words through its
branches (the struct values an `if`, a `match` or a `loop` yields, each
function as often as it is called), which a device's compiler takes
seconds over or fails on, is said to be large, with the structs it
carries and the functions that carry most, each with the calls that
reach it. A scene that publishes Kernels no node uses is told which: each
is compiled and prepared with it all the same. A node with no `domain`
that carries its own output back through an input other than its first
is noted: it runs over that first input, so its state takes that size. A running session (the
Player, `serve`, a replay) says both as they come: a large Kernel with
each version it publishes, the unused ones when the scene loads.

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
  may take (4000 by default); `--small <operations>` the operations a run
  may take, by its elements and its Kernel's typical work, and go to the
  JIT rather than the device (400000 by default; a run of 64 elements or
  fewer goes whatever its work, or, reading what the device made in the
  same evaluation, past 100000 operations; `0` keeps every run on the device, [why](DECISIONS.md#a-small-run-goes-to-the-processor-a-device-is-for-many-elements)).
  A small run rounds as the JIT does, so two forms of one computation a
  replay holds equal to the bit are held with `--small 0` on a GPU.
  `render`, `replay`, `serve`, `graph`, `trace` and `bench` take all three.
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
orla player examples/data/parts/save/scene.osn --allow file.write=examples/data/parts/save/saves
```

## Render frames

```sh
orla render examples/sound/scene.osn --window 640x400 --frames 300 --out /tmp/sound
ffmpeg -framerate 60 -i /tmp/sound/%04d.png -i /tmp/sound/sound.wav -pix_fmt yuv420p /tmp/sound.mp4
orla render examples/sound/scene.osn --frames 600 --out /tmp/sound.wav
```

`orla render <scene.osn> --frames <n>` steps the scene at `--rate` (60
by default) and records each frame. `--out` names where: a directory
(`frames` by default) holds each frame's picture as `<dir>/0000.png`,
... and the sound, block after block, as `<dir>/sound.wav`; a `.wav`
file holds the sound alone. The sound is written at `--sound-rate`
(48000 by default, what Kernels read as `rate()`). The picture and
sound are the outputs the Player would choose. The clock is the frame count over the rate and `async`
runs are waited for after every frame, so a render is the same every
time. The other flags are `run`'s; `--save` writes after the last frame,
which keeps a state trained or settled over many frames as a file a
scene reads back (`input weights: Rows<Float> = "weights.field";`).

## Trace one element

```sh
orla trace examples/thumb/scene.osn --node machine --element 9352 --window 128x128
orla trace examples/life/scene.osn --node life --at 3 4
```

Evaluates what the node reads (at `--time`, zero by default, under
`--param` and `--window`), then runs one element of the node's Kernel on
the interpreter: `--at x y [z]` of a map, or step `--element i` of a scan,
with the steps before it run first. It prints every `let` of the entry
in source order with its type and value, then the result, then each call
the element made (the first 256) with its own `let`s. A `let` in a branch
not taken reads `- (not run)`; a `let` in a loop shows its last
iteration. The Kernel is compiled without the optimizer, so every call
and `let` is present. The trace is of the evaluation after the one last
committed, and commits nothing. An element that fails is traced up to
the failure: what follows it reads `- (not run)`, then `failed:` gives
why, where the operation is written (`file:line:column`), and the `let`s
of the function it failed in. An element found past its budget as it
ends is placed at the loop that ran the most iterations in all. `run` names that place under a failed
evaluation too. The protocol's `trace` is the same as JSON.

## Packages and versions

```sh
orla package packages/text.orlapkg
```

`orla package <dir.orlapkg>` checks a package as `orla check` checks a
document (its manifest, its Kernels compiled, its presets read) and says
what it holds; `orla pack <dir.orlapkg> [<file>]` writes it as one `.orlapkg`
file; `orla unpack <file> <new dir>` reads it back. A scene uses a
package with `use "name.orlapkg";`, by path or by name through
`ORLA_PACKAGES`. `orla version` prints `orla.api`, `orla.language`,
`orla.ir` and `orla.package`; a package needing a newer language or IR
is refused with both numbers.

## Inspect a graph

```sh
orla graph examples/grayscale/scene.osn
```

Prints the flattened graph in canonical form (a placed graph's nodes named
by their path, `title.advances`, so a scene that places one does not read
back from it), the published Kernels with
their generations, and the plan: the groups that execute, in order, with
the producers each absorbed. The nodes holding state (a port reads them
`<~`) are named first, in the order they advance, and under each the
nodes holding state it reads `<-`, as this evaluation made them, and
those it reads `<~`, as the last evaluation committed them; a node
holding none reads what is newest, as a picture of it does. Fusion inlines a pointwise producer into its
only reader over the same domain, at most 64 nodes to a group, so a long
chain fuses into several; `--no-fusion` turns it off. A group a
backend takes Kernel by Kernel but refuses fused (the JIT's stack, a
device's limits) runs apart until one of its Kernels changes, and is
said once with the backend's reason: by `run` and `render`, under a
replay's line, in the Player, beside `bench`'s numbers, and in a
protocol report's `refused`.

## Measure a scene

```sh
orla bench examples/collapse/scene.osn \
    --window 640x400 --backend all --frames 20 --repeat 3 --nodes
```

Steps the scene `--frames` frames `--repeat` times on each backend and
prints the least run, in milliseconds a frame, with the median beside it
and the ratio to the first backend. The first frame of all compiles and
allocates: it is left out of the runs and said apart, with what it took;
every frame waits for the device; the clock
advances by frame count over `--rate`. `--nodes` measures each node too,
waiting after each, so a node placed several times is one line
(`steps/*.spread/* (48)`) and the lines add up to more than the frame.
`--json <file>` writes the measurement; `--compare <file>` reads one back
and prints the earlier number and the change. Over the protocol,
`evaluate` with `measure: true` measures each node of one frame the same
way ([PROTOCOL](PROTOCOL.md)).

## Backends

```sh
orla run examples/showcase/scene.osn --backend gpu --out showcase.ppm
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
orla player examples/life/scene.osn
```

Opens a scene in a window and keeps it live, watching files as `--watch`
does; a changed Kernel compiles on a worker thread and swaps in when
ready, so a republish never stalls a frame. It takes `run`'s
`--library`, `--input`, `--backend`, `--memory`, `--work`, `--small`,
`--allow` and `--grant`, and:

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
Escape; its sticks, triggers and buttons are `stick()` and `button()`,
and `pads()` is how many are connected.

Keys: F6 cycles the preview through the nodes and back to the output;
F5 resets Graph state; F10 shows a line of frame data (frame interval
and rate, evaluate and present times, backend, window and scale, the
port shown, what ran, was reused and was fused, `async` runs,
presentation path and downloads, the sound device's rate, block size and
dry callbacks). Tab, Escape and every other key are the scene's; closing
the window, or Cmd+Q, quits.

## Replay tests

```sh
orla replay examples/life/tests/blinker.replay
orla player --replay examples/voxel/tests/voxel.replay
```

An application's tests are `.replay` files next to its scene, one step a
line, `#` to the end of a line a comment, paths relative to the replay
file. `orla help replay` lists the same forms:

| form | what it does | example |
| --- | --- | --- |
| `scene <path>` | the scene to test, relative to the replay file; required | `scene ../scene.osn` |
| `library <path>` | Kernels beyond the scene's `use` | `library ../kernels/` |
| `alike` | its checks hold two forms of one computation equal word for word, which they are only where every Kernel rounds alike, so `--round-apart` leaves the replay out | `alike` |
| `frame <seconds> [to <seconds> every <seconds>]` | evaluates one frame at this session time; with `to` and `every`, a frame at each step from the first time to the last, both included | `frame 0.0`, `frame 0.1 to 2.0 every 0.1` |
| `settle` | waits for every `async` run to land | `settle` |
| `pointer [<x> <y>] [down \| up]` | moves the pointer, in the output's pixels from its top left (what `pointer()` reads), and presses or releases it | `pointer 20 20 down`, `pointer up` |
| `button <name> down \| up` | left, right, middle, or a pad's south, east, west, north, l1, r1, l3, r3, start, select | `button right down`, `button south up` |
| `stick left \| right \| triggers <x> <y>` | a pad's axis pair | `stick left 0.5 -1` |
| `pads <count>` | how many pads are connected from here on, what `pads()` reads | `pads 1` |
| `wheel <x> <y>` | the scroll for the next frame only | `wheel 0 -1` |
| `window <w> <h> [<scale>]` | the window in pixels, for `window()`, and its scale factor; one when not given | `window 240 135`, `window 240 135 0.25` |
| `key <code> down \| up` | a key by the browser's code: letters and digits by uppercase ASCII, space 32, enter 13, arrows 37 to 40 | `key 67 down` |
| `reset [<node.port>]` | forgets every node's committed state, or one port's | `reset`, `reset step.result` |
| `param <name> <literal>` | sets a graph parameter | `param exposure 1.5` |
| `expect <x> <y> = <r> <g> <b> [<tolerance>]` | the picture's color at x y from its top left, each channel 0 to 1 as the Kernel wrote it; tolerance 0.01 when not given | `expect 11 10 = 1.0 1.0 1.0`, `expect 11 10 = 1.0 1.0 1.0 0.05` |
| `expect <node.port[i][.field]> = <value>... [<tolerance>]` | an element of a node's output, or a field of it (a struct's element is compared a field at a time; an array's element is named by its index, `.cells.2`): ints and bools exactly, floats and vectors within the tolerance; `i` counts along the Field, x fastest, then y, then z | `expect step.result[0].score = 1`, `expect step.result[0].cells.2 = 0` |
| `expect <node.port[i][.field]> < \| <= \| > \| >= <number>...` | every component of it past a bound | `expect fall.result[0].speed > 2.5` |
| `expect count <node.port> = <n>` | how many elements a node's output holds | `expect count ground.result = 6889` |
| `expect placed <node> main \| ordered` | where a node runs: the session's main backend, or ordered work off it; a session with one backend reads both as main | `expect placed ring main`, `expect placed filter ordered` |
| `show <node.port[i][.field]>` | prints an element of a node's output after the last frame, or a field of it, a struct field by field; it checks nothing, and the run reports what it shows | `show step.result[0]`, `show step.result[0].score` |

Nodes an expectation names are pinned before the first frame, so fusion
keeps them; a frame evaluates only what its expectations need. Replays
run together, `orla replay tests/*.replay`, share what their scenes are
made of: the backends are opened once, each document parsed and each
Kernel compiled and prepared once while what it is made from is
unchanged, and each scene's replays play on one session, started over
for each (its state, parameters and pins), as a fresh one would show
them. A large scene's replays run faster together than one by one.

`orla replay tests/*.replay --watch` keeps them so and plays them again as
documents, inputs and replay files change: a change plays only the
replays it reaches (a node their checks read, now or as committed state,
runs a Kernel published again or reads an input read again; the graph
rebuilt otherwise, or the replay's own file changed), those that failed
last first, each said as it ends. A Kernel is published again only when
what it reaches changes, so a function no replay's checks reach plays
none. Each replay played again takes, frame by frame, what the same frame
of its last run computed from what has not changed since, so an edit
computes only what it changes and what reads it. `--round-apart` belongs
to a run before a merge, not to `--watch`.

In the Player, a replay drives the scene at
its own times, prints what it observed at the end, and leaves the window
to the person.

`orla replay --round-apart` runs each replay three ways, some of its
Kernels rounding as another device might (a multiply fused into an add,
a division taken by the reciprocal), chosen by their ids; each way
parts other Kernels. A replay allows what a device's rounding changes
in a picture, so what fails is a value two Kernels each compute and must
hold equal: compute it in one and read it in the other
([why](DECISIONS.md#the-rasterizer-is-a-package-the-native-one-is-its-fast-path)).
An expectation that holds two forms of one computation to each other
word for word, as `examples/vector/tests/tiled.replay` holds the tiled
drawing to the plain one, would fail rounded apart as it should: the two
agree to the bit only where every Kernel rounds alike. Such a replay says
`alike`, and `--round-apart` leaves it out, saying so.
`--no-fusion` runs every node on its own.

`orla replay <file> --out <dir | file.wav>` records the frames the replay
evaluates as `render` records its own: each frame's picture and the
sound, block after block, or the sound alone. `rate()` reads zero in a
replay, as without a device; the WAV is written at `--sound-rate`
(48000 by default), one block for each frame the replay evaluates.

## Import and export

```sh
orla import examples/model/source/crate.glb --out examples/model/assets
orla import examples/skin/source/arm.glb --out examples/skin/assets
orla import song.ogg --out assets   # writes assets/song.wav
orla import examples/score/song.mid --out examples/score/assets
orla import examples/vector/source/garden.svg --out examples/vector/assets
orla import font.ttf --out assets --chars "Hello, world."
```

`orla-import` writes what a scene takes as inputs, into `--out <dir>`
(the source's directory by default):

- glTF or GLB: vertices as rows of `{position, normal, uv}` with node
  transforms baked, triangles as rows of three indices, the base color
  texture as PNG; sparse accessors and JPEG textures are reported and
  skipped. A skin adds `<stem>.weights.field`, `<stem>.skeleton.field`
  and one `<stem>.<name>.clip.field` per animation, sampled at `--rate`
  frames a second (30 by default), per [the rig contract](RIGS.md),
  which the [`skin` package](../packages/skin.orlapkg) poses.
- MP3, OGG Vorbis, FLAC, AAC in MP4, Matroska audio: a WAV at the file's
  rate, one or two channels.
- Standard MIDI: `<stem>.notes.field`, a row a note (start and length in
  seconds, key, velocity 0 to 1, channel).
- SVG: what the [`vector` package](../packages/vector.orlapkg) draws, as four
  Fields: `<stem>.segments.field` (each path's curves cut into segments,
  a stroke as the outline it paints), `<stem>.shapes.field` (colour or
  linear or radial gradient, fill rule, box, in paint order),
  `<stem>.stops.field` and `<stem>.view.field` (the document's size).
  Text, images, patterns, clip paths, masks and filters are reported and
  left out.
- TrueType or OpenType: a font's glyphs, for a scene's Kernels to lay
  out and draw, as three Fields: `<stem>.glyphs.field` (a row a character in
  order of code, its advance in the font's units, its segments and box;
  first the glyph for a character the font lacks), `<stem>.segments.field`
  (the outlines in ems, y up) and `<stem>.metrics.field`. `--chars` names
  the characters to import, every one the font maps by default. Kerning,
  ligatures and shaping are left out.

```sh
orla export crate.vertices.field crate.triangles.field crate.albedo.png --out crate.glb
orla export kick.field --out kick.wav --rate 48000
orla export atlas.field --out atlas.png
orla export song.notes.field --out song.mid --beats 100
```

`orla-export` writes Host files as other software's formats: vertex,
triangle and PNG as a GLB (vertices with a tangent after their uv carry
it as the mesh's tangents; a PNG named `<stem>.normal.png` is the normal
map, `<stem>.occlusion.png` the occlusion, any other the base color;
with a rig's rows, a skin and its animations,
`--rate` 30 by default; `--filter nearest` samples the PNG texel by texel,
as pixel art wants, where `linear` blends; `--joints a,b,...` names the
joints in the skeleton's order); rows of one or two floats as a WAV (`--rate` 48000
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
connects from this machine, or from an origin `--allow-origin` names
(`null` for a page opened from a file). `--window` is the window evaluations see unless
they name one. `orla-player --serve` serves the same protocol from the
window a person is watching. The operations are in
[PROTOCOL](PROTOCOL.md).

## In your own process

From a release, whose library is `lib/` and header `include/`:

```sh
cc bridges/c/hello.c -Iinclude -Llib -lorla_ffi -Wl,-rpath,lib -o hello
./hello examples/text/scene.osn jit
```

From the source:

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
```

Each bridge's README says how to open it. From the source, `cargo test
-p orla --test bridges` runs web through node and Chrome and Godot
through `godot -s`, and skips, saying so, what has no tool installed.

## WebAssembly

A release carries both: `orla-<version>-wasm32-wasip1.wasm`, the `orla`
command for any WASI runtime, and `orla-<version>-web.tar.gz`, Orla in a
page with the web bridge, the examples and the packages:

```sh
wasmtime run --dir . orla-<version>-wasm32-wasip1.wasm replay examples/ant/tests/ant.replay --backend cpu
cd orla-<version>-web && node bridges/web/serve.mjs   # then http://127.0.0.1:7789/bridges/web/play.html?scene=examples/voxel/scene.osn
```

From the source:

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
