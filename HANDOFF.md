# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits
on `feat/first-proof` are authorized; do not merge or push. Commit as the
configured identity. Preserve both architectural bibles verbatim (`Orla Architecture
Plan.md`, `Orla — Minimal Core Architecture.md`); their handle lists stay as
written while the code has one `Handle`.

The previous implementation at `../bak/orla` is a read-only reference.

Layering as decided: the runtime crates and the `orla` CLI take no external
dependencies; the interface layer (L8: `cli`, `player`, `import`, `export`,
`jit`, `ffi`) is the only place that may (player: winit, softbuffer, cpal,
gilrs; gpu: wgpu; import: symphonia; export: none; jit: cranelift 0.131,
the newest that builds on Rust 1.93; ffi: none). `jit` and `ffi` are the
two crates with `unsafe`. Crates are named by what they are, not by
layer. Bridges live in `bridges/<program>/` in that program's language, outside
the workspace, speaking only the protocol and the Host's file formats.

Decisions not recoverable from code: lossy audio is decoded (importer) but
never encoded (WAV is the way out; ffmpeg compresses); JPEG is not decoded
(convert to PNG); interlaced PNG is refused; `.wav` saves at 48 kHz; a node's
`times` is structural, so a param it names lowers the document again; `sample`
stays in `compute` while functions with Field parameters use `at`/`at_linear`;
text is a convention of the shared library, not a contract of the runtime:
`examples/shared/text.orla` declares the glyph row and the drawing, and the
font atlas beside it is baked from rows of bits in `font.orla` (a face is
replaced by re-baking or by another PNG in the same 6x8 cells); a scatter's
order is its determinism rule (sources applied one after another in element
order, floats included), which the GPU honours by applying the pairs in one
invocation after a parallel map, rather than with atomics: exact and slow
for very many sources, and a place to optimise later without changing any
result (int `sum`, `min`, `max`, and `last` by index are order-free); the
the CPU compiles a module to word code on every execution because that
was measured at 2 to 40 microseconds per Kernel against milliseconds of
running (voxel, raymarch, life), so no compiled program is cached across
frames; a headless session's window is the session's (`serve --window`,
the `window` op) unless an `evaluate` names one, and the Player's is the
person's;
scatter's `min` and `max` order floats by their bits' total order on both
backends so the GPU may use atomics, and a float `sum` is a bitonic sort
of (target, source) followed by a per-target fold in source order, all
sort steps in one compute pass (dispatches in a pass see each other's
writes in order; a pass per step cost ten times more), which is exact
and, at a hundred thousand sources into 320x240, under 2 ms on the GPU
against 38 on the CPU, where the walk in one invocation took about 15;
the
window is context under `input.read` rather than a capability of its own
(its size is in physical pixels, its scale factor is what a bar laid out
in logical pixels divides by), the Player reports a default 960x540 window
before its own opens, and a resize needs no structural step because the
plan is drawn every evaluation and the extent is in the cache key; the
listener tells a WebSocket from a line client by the first four bytes
(`GET `) and never serves a page, so a bridge's files are opened from
disk, and `orla.js` is a classic script because browsers refuse module
imports between file URLs; a scan's step loop on the GPU has a
`storageBarrier()` between steps because the Metal compiler hoisted a
step's first memory load out of the loop (every step read the starting
value; the CHIP-8 escaped it by writing before reading); the JIT is on Cranelift 0.131 (the newest that
builds on Rust 1.93), returns every function's result through a
caller-provided stack slot (arm64 refuses a struct's worth of return
registers), keeps `unsafe` to the one crate that calls the code it made
(its own `[lints]` instead of the workspace's `forbid`), and prepares one
compiled module per Kernel version, freed with it; a reduction's `merge`
is compiled with its two values passed through pointers (its words vary
by Kernel, a native call from Rust cannot), and the fold and a scatter's
apply are the CPU crate's loops so the three backends stay one rule; the
C ABI (`orla-ffi`) is blocking and single-threaded per session, its
strings owned by the session until the next call and a Field's words
until the next evaluate, request or close (the Host may release them),
its header written by hand rather than generated so no build dependency
comes in, and the whole protocol reachable through one `orla_request`
so the fast path stays ten functions; a trace is of the evaluation
that would follow the one it runs first (the node's `<~` inputs are the
state just committed) because the inputs of the evaluation just done are
gone once it commits, and the trace commits nothing itself.

Before a commit run, in this order: `cargo fmt --all`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo test --workspace` (about 40 s;
the example-replay test is the slow one), and every replay on the three backends:
`for f in examples/*/tests/*.replay; do for b in gpu cpu jit; do
./target/release/orla replay $f --backend $b; done; done` (about 20 s after
`cargo build --release`). Tests in one binary run in parallel: a test that
writes files takes a directory of its own (`scratch_in`, or a named scratch).

The Blender bridge is tested headless: `orla serve examples/mesh/scene.osn
--listen 127.0.0.1:0` (port on stderr) then
`/Applications/Blender.app/Contents/MacOS/Blender -b --python
bridges/blender/test_bridge.py -- --port <port> --out <dir>`. The web
bridge is tested by `node bridges/web/test.mjs` after `cargo build
--release`: it serves the text and voxel examples itself and drives
`bridges/web/test.html` in a headless Chrome over the DevTools protocol
(the `--dump-dom` shortcut does not wait for asynchronous work in the
new headless mode, which is why a runner exists). The C ABI is tested by
`cargo test -p orla-ffi` (in the workspace run) and by hand with
`cc bridges/c/hello.c -Icrates/ffi/include -Ltarget/release -lorla_ffi`
then `DYLD_LIBRARY_PATH=target/release ./a.out examples/text/scene.osn
jit`, which prints the picture's size, a pixel before and after a
publish. The Godot bridge is
tested by `godot --headless --path bridges/godot -s test.gd` after
`cargo build --release` (`/usr/local/bin/godot`, 4.6): the script spawns
the text, voxel and mesh sessions through `OS.execute_with_pipe`, reads
each port from stderr, and ends with `RESULT: ok`; its scripts reach
each other by `preload`, since a headless `-s` run has no global class
cache. The user's own
Blender runs an MCP add-on on 127.0.0.1:9876 that executes Python sent as
`{"type":"execute","code":...,"strict_json":false}` terminated by `\0`,
answering the same way; the bridge was driven live through it (push with Live,
pull, Watch, publish, `param level`). A play session's Player and scratch
files may be left running or gone; nothing of it is in the repository.

## Next

The machine substrate is proven and closed for now: a machine is a
`scan` over memories (`put`, also from functions with a Field
parameter), with `>>>`, `add_wrap`/`sub_wrap`/`mul_wrap` and hexadecimal
literals for its ALU, stepped by the JIT; the
[CHIP-8](examples/chip8/README.md) and the
[Thumb core](examples/thumb/README.md) (every format of the ARM7TDMI's
16-bit set, 19 million instructions a second on the JIT, above a GBA's
clock) are the proofs. A whole emulator (GBA or otherwise) is a project
of its own, not a next step of this repository. The JIT now covers every
mode (a reduction folds with its compiled `merge` in the interpreter's
tree, a scatter applies with the CPU crate's combine rule), so
`--backend jit` never falls back to the interpreter. Back to the
platform. The third bridge, Godot, is in (`bridges/godot`: an
`OrlaView` Control as the window, params as controls, an `ArrayMesh`
from rows, tested headless with the sessions spawned by Godot itself),
and so is the in-process door: `orla-ffi`, a C ABI over the session
(`crates/ffi/include/orla.h`, `bridges/c/hello.c` links it), the
Fields read where they lie.

1. The Kernel language's gaps. The debugging surface is in: `orla
   trace` and the protocol's `trace` run one element or one scan step of
   a node on the interpreter and show every `let` with its value, the
   ones in branches not taken marked; `run --profile` and the report's
   `timings` time each node. Calls are not followed (a called function's
   frame is gone when the entry returns) and a loop's bindings show the
   last iteration: two places to deepen if wanted. Next the language's
   ceiling where it was felt writing the CHIP-8 and Thumb cores: a
   `match` on an `Int` (a decoder is one, not an `if` chain; the GPU
   emits a `switch`, the JIT a jump table), an outer loop's carried
   value readable inside a nested loop, and whatever else those two
   examples show once rewritten with them. Not taken up: 64-bit ints
   (WGSL has none), strings (text is rows of codes), dynamic sizes
   (a graph resizes by params), `put` outside `scan` (determinism).
2. What a session of play turns up: the voxel editor and the showcase
   were rated on interactivity; the JIT closed the CPU gap, the GPU scan
   is one thread (a machine belongs on the JIT).

Left as is: the interpreter reports an arithmetic failure in a scan at
`(index, 0)` and so does the JIT; the JIT's parity is against the CPU's
words, so a change to the interpreter's rounding needs a change here.

## Open

Not hand-tested: the Player's real right and middle buttons and wheel,
gamepad hardware (none connected), and a live resize of a window-sized
scene in the Player (the Player was opened on the voxel editor and
presented; the resize mechanism is covered by the graph test, not by a
hand on the window's corner). The web bridge's page was driven in a real
Chrome as well as headless: the text example's title republished from
its editor and the voxel editor filled a Retina canvas at 24 ms a frame
and took a click; a browser refuses `file://` pages through the Chrome
extension, so the directory was served with any static file server for
that (`python3 -m http.server`), which the page does not need otherwise.
The Godot bridge's window was opened (no script errors) but not
clicked through by hand; its demo scene is driven by the headless test
instead. The Blender bridge ran on Blender 5.2
only; Watch polls (default 0.5 s) and reads whole rows each tick. The file
watch polls every 200 ms and can observe a half-written file; the next write
is picked up. The CPU interpreter is the reference; it compiles to
word-addressed code and runs the raymarch scene at 400x260 in about 0.35 s
and a voxel frame at 960x540 in about 1.3 s on ten cores, so it is still
not interactive for those; the JIT runs the same frames in 22 ms and
92 ms. The worker is one FIFO thread.
Audio pacing is one block behind the ring. Native manifests print element
types as anonymous structs.
