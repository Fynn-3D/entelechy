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
`jit`) is the only place that may (player: winit, softbuffer, cpal, gilrs;
gpu: wgpu; import: symphonia; export: none; jit: cranelift 0.131, the
newest that builds on Rust 1.93, and the one crate with `unsafe`). Crates are named by what they are, not by
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
imports between file URLs; the JIT is on Cranelift 0.131 (the newest that
builds on Rust 1.93), returns every function's result through a
caller-provided stack slot (arm64 refuses a struct's worth of return
registers), keeps `unsafe` to the one crate that calls the code it made
(its own `[lints]` instead of the workspace's `forbid`), and prepares one
compiled module per Kernel version, freed with it.

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
new headless mode, which is why a runner exists). The user's own
Blender runs an MCP add-on on 127.0.0.1:9876 that executes Python sent as
`{"type":"execute","code":...,"strict_json":false}` terminated by `\0`,
answering the same way; the bridge was driven live through it (push with Live,
pull, Watch, publish, `param level`). A play session's Player and scratch
files may be left running or gone; nothing of it is in the repository.

## Next

Machines on the JIT, which now compiles map and scan Kernels (a scan's
`init`, its carried words and index outside loops, `put` into working
copies of the memories that become the outputs; every example replay
passes on `--backend jit` bit for bit against the CPU, the raymarch,
voxel and showcase scenes twenty to thirty-five times faster and four
million CHIP-8 steps in 0.18 s against 1.67 s). Two directions, in order:

1. A GBA as the flagship machine: an ARM7TDMI core as one scan with
   memories for the BIOS, work RAM, VRAM, palettes and OAM, a PPU as map
   Kernels over VRAM. Sixteen million cycles a second is about the
   interpreter-free step rate the CHIP-8 measured (twenty-two million
   trivial steps a second), so a GBA needs each step to be several
   instructions and the PPU on the GPU; measure a scan with a real
   fetch-decode-execute before committing to full speed.
2. Reductions and scatters on the JIT: the map part compiles as today,
   the fold and the in-order apply follow the CPU's, so a scene that
   mixes them no longer switches to the interpreter mid-graph.

Done on the way: `put` inside functions with a Field parameter (a store
helper for a core, refused in `init` even through a call) and `>>>`,
the logical shift right, on all three backends.

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
that (`python3 -m http.server`), which the page does not need otherwise. The Blender bridge ran on Blender 5.2
only; Watch polls (default 0.5 s) and reads whole rows each tick. The file
watch polls every 200 ms and can observe a half-written file; the next write
is picked up. The CPU interpreter is the reference; it compiles to
word-addressed code and runs the raymarch scene at 400x260 in about 0.35 s
and a voxel frame at 960x540 in about 1.3 s on ten cores, so it is still
not interactive for those; the JIT runs the same frames in 22 ms and
92 ms. The worker is one FIFO thread.
Audio pacing is one block behind the ring. Native manifests print element
types as anonymous structs.
