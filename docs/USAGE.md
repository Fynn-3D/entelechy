# Usage

**Answers:** which command does what, with a line you can run.

`entelechy` prints every command with no arguments, and `entelechy help
<command>` every flag of one; this page is the commands with a line to
run. `entelechy player`, `entelechy import` and `entelechy export` run
`entelechy-player`, `entelechy-import` and `entelechy-export` beside
`entelechy` with the same arguments; their flags are below.

In a release, `entelechy` is `bin/entelechy`, and the paths below are relative to
the unpacked directory. From the source, `cargo build --release` puts
every program in `target/release/`, and `cargo run --release --` runs
`entelechy`:

```sh
cargo build --workspace
cargo test --workspace
```

## Start here

```sh
entelechy new hello
entelechy player hello/scene.esn
```

`entelechy new <dir>` writes a directory that runs: a Kernel, the scene
that places it, a replay that checks what it draws, and a README saying
what to change first; `--kind picture|sound|game` picks the start. Edit
the Kernel while the Player is open; a Kernel that does not compile
leaves the running one alone and says why.

## Syntax and API

```sh
entelechy help              # every layer's syntax and API, and the commands
entelechy help smoothstep   # one name in full: a builtin's signatures and capability
entelechy help window       # every item of a name: the builtin, the replay line, the operation
entelechy help publish --json
```

Every answer comes from the table the part it describes executes; the
same text is [REFERENCE](REFERENCE.md). The language is in
[DOCUMENTS](DOCUMENTS.md), whole programs in `examples/`.

## Check a Kernel

```sh
entelechy check --ir examples/grayscale/kernels/grayscale.ent
```

Compiles `.ent` or `.esn` files and prints their manifests, or every
fault at `file:line:column`. A scene's nodes are held to their Kernels'
ports as an evaluation binds them. It also notes a Kernel large for a
device's compiler, Kernels no node uses, and nodes whose elements past
their first input are never run; a running session says the same as it
publishes. `--ir` prints the validated IR as a backend runs it.

## Run a scene

```sh
entelechy run examples/grayscale/scene.esn --out output.ppm
entelechy run scene.esn --save steps/8.spread/6.result=rock.field
entelechy-player editor.esn --input disk=~/code/thing/ --output done=~/code/thing/ \
  --allow file.write=~/code/thing
```

Evaluates a scene once. `--save <node.port>=<file>` writes a node's
output (a node of a placed graph by its path, the port after the last
dot), which is how an asset is made; rows go out as a `.csv` or `.json`
table a scene reads back. `--input` and `--output` place a
scene's inputs and outputs elsewhere, its text unchanged. Writes land
only where `--allow` says; `--profile` prints each node's time, backend
and transfers, waiting for the device after each, and each Kernel's
preparing: how long its compiling and its building on the device took.

## Render frames

```sh
entelechy render examples/sound/scene.esn --window 640x400 --frames 300 --out /tmp/sound
ffmpeg -framerate 60 -i /tmp/sound/%04d.png -i /tmp/sound/sound.wav -pix_fmt yuv420p /tmp/sound.mp4
entelechy render examples/sound/scene.esn --frames 600 --out /tmp/sound.wav
```

Steps the scene at `--rate` and records each frame's picture and the
sound. The clock is the frame count over the rate and `async` runs are
waited for every frame, so a render is the same every time; `--save`
after the last frame keeps a trained or settled state as a file.

## Trace one element

```sh
entelechy trace examples/machines/parts/thumb/scene.esn --node machine --element 9352 --window 128x128
entelechy trace examples/life/scene.esn --node life --at 3 4
```

Runs one element of a node's Kernel on the interpreter and prints every
`let` with its type and value, the result, and each call it made; an
element that fails is traced to the failure, with where it is written.
It commits nothing. The protocol's `trace` is the same as JSON.

## Packages and versions

```sh
entelechy package packages/ui.entpkg
```

Checks a package and says what it holds; `entelechy pack` writes it as
one `.entpkg` file and `entelechy unpack` reads it back. A scene uses a
package by path or by name through `ENTELECHY_PACKAGES`. `entelechy
version` prints the API, language, IR and package versions.

## Inspect a graph

```sh
entelechy graph examples/grayscale/scene.esn
```

Prints the flattened graph, the published Kernels with their
generations, the state in the order it advances, and the plan: the
groups that execute, in order, with the producers each absorbed, and
under each group the producers only it reads that run apart from it,
each with why (it sleeps, its work is budgeted, it reads them by
address). `--window <w>x<h>` plans what reads the window at that size.

## Measure a scene

```sh
entelechy bench examples/collapse/scene.esn \
    --window 640x400 --backend all --frames 20 --repeat 3 --nodes
```

Prints the least run in milliseconds a frame, the median, and the ratio
to the first backend; the first frame is said apart. The frames overlap
the device as a window's do; a device's frame is split into its own work
and its waits for the device, and the nodes that waited are named.
`--nodes` times each node as a window's frame-data line does: on a
device by its own clock beside the call on the frame's thread, nothing
waiting for either (waiting after each node only where the device keeps
no clock); `--json` and `--compare` keep a measurement and compare
against it.

## Backends

```sh
entelechy run examples/showcase/scene.esn --backend gpu --out showcase.ppm
```

- `cpu` is the interpreter, the reference, and what `trace` walks.
- `jit` compiles the IR to native code through Cranelift, bit for bit
  the interpreter's results; it is the default.
- `gpu` runs the IR as WGSL compute through wgpu and keeps Fields on the
  device; small runs and what the device cannot take go to the JIT, said
  with why.
- `best` is the GPU where there is a device, else the JIT; the Player's
  default.

## Hot reload

```sh
entelechy run examples/grayscale/scene.esn --out output.ppm --watch
```

Save the scene or any document it uses: the Kernels whose text changed
republish and what they affect runs again. A Kernel that does not
compile is reported and the previous version keeps running.

## Player

```sh
entelechy player
entelechy player examples/life/scene.esn
entelechy-player editor.esn --folder disk,checks,defined,disk_ops --folder-allow file.write,file.delete
```

Opens a scene in a window and keeps it live; a changed Kernel compiles
off the frames and swaps in when ready, and any compile or evaluation
fault is shown directly on the canvas until the next working frame.
Without `<scene.esn>` (or `--replay`), it opens an empty `0 Hz` resting
stage: `Cmd+O` (`Ctrl+O` elsewhere, when `--folder` is not set) chooses a
`.esn` or `.replay` file, and dropping a `.esn`, `.replay` or project
folder onto the window switches to it in place over the same warm
backend. It takes `run`'s `--library`,
`--input`, `--output`, `--backend`, `--memory`, `--work`, `--small`,
`--allow` and `--grant`, and:

- `--scale <n>`, `--window <w>x<h>`: the initial magnification and size.
- `--no-audio` leaves the sound card closed.
- `--show-while-building` opens the window before the device has built
  what only shows, drawing it on the processor meanwhile.
- `--hold-pointer` locks and hides the system cursor once clicked, keeping
  `pointer()` inside the output; `--lock-pointer` locks and hides it with
  unbounded relative motion (for 360-degree camera aim; Escape releases it).
- `--folder <name,...>` lets Cmd+O choose a folder for those directory
  inputs and outputs, allowed what `--folder-allow` says.
- `--replay <file>` plays a replay in the window.
- `--serve`, `--listen <host:port>`, `--allow-origin <origin>` serve the
  protocol from the window.

The window shows the first picture given no device; what the scene
gives `to sound` the sound card plays, and what it gives `to pointer` is
the system's pointer over the window, pointing with what it gives `to
pointer_tip`. A gamepad is
keys too: the D-pad and left stick are the arrows, South Z, East X,
Start Enter, Select Escape. F5 resets state, F6 cycles the preview
through the nodes, F8 holds the scene's time and F9 or F7 steps it, F10
shows the frame's data (`frame`, `cpu`, `wait`, `gpu`, nodes, memory,
the nodes that waited for the device, and the busiest nodes on it). Every other key is the scene's.

## Replay tests

```sh
entelechy replay examples/life/tests/blinker.replay
entelechy replay tests/*.replay --watch
entelechy player --replay examples/voxel/tests/voxel.replay
```

An application's tests are `.replay` files next to its scene, one step a
line, `#` a comment, paths relative to the replay file:

| form | what it does | example |
| --- | --- | --- |
| `scene <path>` | the scene to test, relative to the replay file; required | `scene ../scene.esn` |
| `library <path>` | Kernels beyond the scene's `use` | `library ../kernels/` |
| `allow file.write <dir> \| allow file.delete <dir>` | lets the scene's outputs write (or remove and rename) files under the directory, relative to the replay file, as `--allow` does; what they write is kept for the run and never reaches the disk, so every run starts from the same files; with or without it, what the scene saved for itself before the run (an output's file an input reads, with a fallback) is not read until the run writes it | `allow file.write ../laws/` |
| `alike` | its checks hold two forms of one computation equal word for word, which they are only where every Kernel rounds alike, so `--round-apart` leaves the replay out | `alike` |
| `frame <seconds> [to <seconds> every <seconds>]` | evaluates one frame at this session time; with `to` and `every`, a frame at each step from the first time to the last, both included | `frame 0.0`, `frame 0.1 to 2.0 every 0.1` |
| `settle` | waits for every `async` run to land | `settle` |
| `pointer [<x> <y>] [down \| up]` | moves the pointer, in the output's pixels from its top left (what `pointer()` reads), and presses or releases it | `pointer 20 20 down`, `pointer up` |
| `button <name> down \| up` | left, right, middle, or a pad's south, east, west, north, l1, r1, l3, r3, start, select | `button right down`, `button south up` |
| `stick left \| right \| triggers <x> <y>` | a pad's axis pair | `stick left 0.5 -1` |
| `pads <count>` | how many pads are connected from here on, what `pads()` reads | `pads 1` |
| `wheel <x> <y>` | the scroll for the next frame only | `wheel 0 -1` |
| `window <w> <h> [<scale>]` | the window in pixels, for `window()`, and its scale factor; one when not given | `window 240 135`, `window 240 135 0.25` |
| `type "<text>"` | characters typed, what `typed()` reads on the next frame (32 a frame, the rest on the frames after); `\"`, `\\`, `\n` and `\t` inside the quotes, `#` among them no comment | `type "你好, Quill"` |
| `clipboard "<text>"` | the clipboard holding this text from here on, as a person copying elsewhere leaves it: what an input `= clipboard` reads; the same escapes as `type` | `clipboard "from elsewhere"` |
| `key <code> down \| up` | a key by the browser's code, as `key()` reads it: letters and digits by uppercase ASCII, space 32, enter 13, arrows 37 to 40, the rest in `help key` | `key 67 down` |
| `reset [<node.port>]` | forgets every node's committed state, or one port's | `reset`, `reset step.result` |
| `param <name> <literal>` | sets a graph parameter | `param exposure 1.5` |
| `expect <x> <y> = <r> <g> <b> [<tolerance>]` | the picture's color at x y from its top left, each channel 0 to 1 as the Kernel wrote it; tolerance 0.01 when not given | `expect 11 10 = 1.0 1.0 1.0`, `expect 11 10 = 1.0 1.0 1.0 0.05` |
| `expect <node.port[i][.field]> = <value>... [<tolerance>]` | an element of a node's output, or a field of it (a struct's element is compared a field at a time; an array's element is named by its index, `.cells.2`): ints and bools exactly, floats and vectors within the tolerance; `i` counts along the Field, x fastest, then y, then z | `expect step.result[0].score = 1`, `expect step.result[0].cells.2 = 0` |
| `expect <node.port[i][.field]> < \| <= \| > \| >= <number>...` | every component of it past a bound | `expect fall.result[0].speed > 2.5` |
| `expect count <node.port> = <n>` | how many elements a node's output holds | `expect count ground.result = 6889` |
| `expect placed <node> main \| ordered` | where a node runs: the session's main backend, or ordered work off it; a session with one backend reads both as main | `expect placed ring main`, `expect placed filter ordered` |
| `show <node.port[i][.field]>` | prints an element of a node's output after the last frame, or a field of it, a struct field by field; it checks nothing, and the run reports what it shows | `show step.result[0]`, `show step.result[0].score` |

Replays run together share their backends, documents and
compiled Kernels, so a large scene's run faster so than one by one.
`--watch` plays again only the replays a change reaches, each computing
only what changed. `--round-apart` runs each replay three ways, some
Kernels rounding as another device might: what fails is a value two
Kernels each compute and must hold equal, so compute it in one
([why](DECISIONS.md#the-rasterizer-is-a-package-the-native-one-is-its-fast-path));
a replay that holds two forms equal word for word says `alike` and is
left out. `--out` records a replay's frames as `render` does.

## Import and export

```sh
entelechy import examples/mesh/parts/model/source/crate.glb --out examples/mesh/parts/model/assets
entelechy import song.ogg --out assets   # writes assets/song.wav
entelechy import examples/sound/parts/score/song.mid --out examples/sound/parts/score/assets
entelechy import examples/vector/source/garden.svg --out examples/vector/assets
entelechy import font.ttf --out assets --chars "Hello, world."
entelechy export crate.vertices.field crate.triangles.field crate.albedo.png --out crate.glb
entelechy export kick.field --out kick.wav --rate 48000
entelechy export song.notes.field --out song.mid --beats 100
entelechy export x.glyphs.field x.segments.field x.metrics.field --out x.woff --name "X Sans" --chars-file words.txt
entelechy export examples/voxel/scene.esn --out voxel.html   # one page that plays the scene
```

`entelechy-import` writes what a scene takes as inputs into `--out`:

- glTF or GLB: vertices `{position, normal, uv}`, triangles, the base
  color as PNG; a skin adds weights, skeleton and clips at `--rate` (30),
  per [the rig contract](RIGS.md). Each triangle's part, the node's
  primitive it came from, is a row of `<stem>.parts.field` (one `Int`),
  and `<stem>.parts.json` names each part by its node, mesh and material,
  so a scene can tell a model's skin from its hair.
- MP3, OGG Vorbis, FLAC, AAC, Matroska audio: a WAV.
- MIDI: a row a note (start, length, key, velocity, channel).
- SVG: the segments, shapes, stops and view the
  [`vector` package](../packages/vector.entpkg) draws.
- TrueType or OpenType: glyphs, outlines and metrics for `--chars`, or
  every character; a scene may also read a font itself with
  `node glyphs: font "x.ttf" { codes <- wanted.result; }`.
- PNG, PPM, or PGM images: hardware block-compressed textures (`.ktx2` by
  default, or `--container dds`) with `--format bc7|bc5|bc4|bc1` (default
  `bc7` for color, `bc4` for grayscale `.pgm`). Scenes read `.ktx2` and
  `.dds` directly into `Grid<... as unorm8>` or streamed `Volume<... as unorm8>`
  `fields` nodes with zero CPU decompression on hardware supporting
  `TEXTURE_COMPRESSION_BC`.

`entelechy-export` writes Host files as other software's formats:
vertices, triangles and PNGs as a GLB (`<stem>.normal.png`,
`.occlusion.png` and `.orm.png` as those maps, a rig's rows as a skin;
`--filter nearest`, `--joints`, `--double-sided`), rows of floats as a
WAV, a grid as a PNG, notes as MIDI, a font's glyphs, segments and
metrics as TrueType, OpenType (CFF) or WOFF (`--name` its family,
`--chars` or `--chars-file` the characters it keeps). What the importer
read is written back bit for bit, but a font: its segments come back as
curves through their ends, within a font unit.

A scene (`.esn`) is exported as a standalone binary bundle, `--out
<name>.bundle`, or as one page, `--out <name>.html`: a single
file that is the Player in a page, Entelechy's slim playback runtime for a
page, and a compiled binary bundle of the scene's IR, lowered graph, input
Fields and compact font contours (with no host paths or local save files). It
opens from the disk with nothing beside it and fetches nothing. `--web` names
where Entelechy for the web is (the directory holding `bridges/web` and
`target/page`: a web release, or a checkout that has built it as
[WebAssembly](#webassembly) says; the current directory by default), and
`--backend` (`gpu` or `jit`) what runs the scene in the page (by default the
page chooses the GPU when WebGPU or WebGL 2.0 is available, else the JIT). `--name` sets
the page's title (otherwise derived from the scene's stem or project folder). `--hold-pointer`
captures the browser cursor on click and keeps `pointer()` inside the output;
`--lock-pointer` captures it and steps `pointer()` without bound (Escape
releases either). The page uses WebGPU (with JSPI) or synchronous WebGL 2.0 for
the GPU backend, or runs on WebAssembly JIT alone (`entelechy_bundle_wasm`
emits each Kernel's position-independent WebAssembly module for hosts that
precompile static `.wasm` modules). A
file the scene only reads later, by a name it computes, is not in the page.
Each `font` node whose `codes` come from static inputs alone keeps only those
characters; `--chars` and `--chars-file` (each may be given again) name the
characters to keep instead when a scene computes them as it runs, and a
character not among them is then not drawn.

The same scene (`.esn`) or precompiled bundle (`.bundle`) is exported as a
native desktop application or a mobile project, with persistent scene saves
(`output ... to "<slot>";` paired with `input ... = "<slot>" else ...;`)
carried in the bundle and stored automatically by each host:

- `--out <name>.app` writes a macOS application bundle, placing `app.entapp` in
  `Contents/Resources/` and sealing `Contents/_CodeSignature/CodeResources` and
  the Mach-O `entelechy-runner` (`--runner <file>`, else beside
  `entelechy-export`) so Gatekeeper's strict bundle signature check holds even
  when packed outside macOS (including WASI).
- `--out <name>.exe` (with `--runner <entelechy-runner.exe>`) or `--out
  <name>.bin` appends the `Standalone` envelope and a 24-byte trailer to a
  prebuilt Windows PE or Unix `entelechy-runner` binary, producing one
  executable file with no compiler or sidecar files (`--out <name>.entapp`
  writes the `Standalone` envelope on its own). `--window <w>x<h>` fixes the
  initial logical window size; `--app-id` sets a `.app`'s bundle identifier
  (`run.entelechy.<slug>` by default); `entelechy-runner --save-dir <dir>`
  overrides the user save folder.
- `--out <dir>.ios` and `--out <dir>.android` write an Xcode project
  (`bridges/ios`, Metal + `AVAudioEngine`, no runtime JIT) or an Android NDK
  `NativeActivity` project (`bridges/android`, `ANativeWindow` + `AAudio`)
  over `entelechy.h` and `scene.bundle`; `--lib` copies a prebuilt
  `libentelechy_ffi.a` or `libentelechy_ffi.so` into the project.

## Creation protocol

```sh
entelechy serve examples/life/scene.esn
```

Agents drive a session with JSON lines on stdin and stdout; `--listen
<host:port>` also serves every connection, upgrading HTTP to a
WebSocket, from this machine or an origin `--allow-origin` names.
`entelechy-player --serve` serves the same protocol from the window a
person is watching. The operations are in [PROTOCOL](PROTOCOL.md).

## In your own process

```sh
cc bridges/c/hello.c -Iinclude -Llib -lentelechy_ffi -Wl,-rpath,lib -o hello   # a release
./hello examples/text/scene.esn jit
cargo build --release -p entelechy-ffi                                          # the source
cc bridges/c/hello.c -Icrates/interface/ffi/include -Ltarget/release -lentelechy_ffi -o hello
DYLD_LIBRARY_PATH=target/release ./hello examples/text/scene.esn jit
```

`libentelechy_ffi` is behind the C header
[`entelechy.h`](../crates/interface/ffi/include/entelechy.h): a host opens
a scene from a path or from files it hands in, evaluates it under its
own input, reads an output in place, and sends protocol lines through
`entelechy_request`. [`bridges/c`](../bridges/c/README.md) is the example.

## Bridges

```sh
entelechy player --serve --listen 127.0.0.1:7788 examples/mesh/scene.esn
```

`bridges/blender`, `bridges/web` and `bridges/godot` speak the protocol
over `--listen`; each README says how to open it. `cargo test -p
entelechy --test bridges` runs them where their tools are installed.

## WebAssembly

```sh
wasmtime run --dir . entelechy-<version>-wasm32-wasip1.wasm replay examples/ant/tests/ant.replay --backend cpu
cd entelechy-<version>-web && node bridges/web/serve.mjs   # then http://127.0.0.1:7789/bridges/web/play.html?scene=examples/voxel/scene.esn
```

From the source:

```sh
cargo build --release -p entelechy --target wasm32-wasip1
cargo build --profile small -p entelechy-ffi --target wasm32-unknown-unknown
wasm-bindgen --target web --keep-lld-exports --no-typescript \
  --remove-name-section --remove-producers-section --out-dir target/page \
  target/wasm32-unknown-unknown/small/entelechy_ffi.wasm
cargo build --profile small -p entelechy-ffi --no-default-features --target wasm32-unknown-unknown
wasm-bindgen --target web --keep-lld-exports --no-typescript \
  --remove-name-section --remove-producers-section --out-name entelechy_single \
  --out-dir target/page \
  target/wasm32-unknown-unknown/small/entelechy_ffi.wasm
node bridges/web/serve.mjs     # then http://127.0.0.1:7789/bridges/web/index.html
```

The `small` profile builds Entelechy for a page small (for size, one unit
linked whole, no unwinding, no names): a page downloads `entelechy_ffi`, and
a scene exported as one page carries `entelechy_single` (the playback runtime
without the compiler, documents or protocol).

Under WASI, `entelechy` runs scenes on the interpreter and `serve`
answers on stdin and stdout. In a page, Entelechy is the C ABI with
`wasm-bindgen`'s glue: its documents are files the page hands in, its
JIT writes WebAssembly the page loads, and its GPU backend runs on the
browser's WebGPU (Chrome and Edge 137 and later). `play.html` is the
Player in a page, `?scene=` or `?replay=`.

## Where the contracts live

- Documents: [DOCUMENTS](DOCUMENTS.md), [the frontend](../crates/frontend/language/src/lib.rs), [scene lowering](../crates/frontend/scene/src/lib.rs).
- Capabilities and types: [the ABI](../crates/core/abi/src/lib.rs).
- IR and validation: [the IR](../crates/core/ir/src/lib.rs), [its validator](../crates/core/ir/src/validate.rs).
- Publication: [the Kernel Runtime](../crates/core/kernel/src/lib.rs).
