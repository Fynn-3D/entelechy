# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits
on `feat/first-proof` are authorized; do not merge or push. Commit as the
configured identity. Preserve both architectural bibles verbatim (`Orla Architecture
Plan.md`, `Orla — Minimal Core Architecture.md`); their handle lists stay as
written while the code has one `Handle`.

The previous implementation at `../bak/orla` is a read-only reference.

Layering as decided: the runtime crates and the `orla` CLI take no external
dependencies; the interface layer (L8: `cli`, `player`, `import`, `export`) is
the only place that may (player: winit, softbuffer, cpal, gilrs; gpu: wgpu;
import: symphonia; export: none). Crates are named by what they are, not by
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
window is context under `input.read` rather than a capability of its own
(its size is in physical pixels, its scale factor is what a bar laid out
in logical pixels divides by), the Player reports a default 960x540 window
before its own opens, and a resize needs no structural step because the
plan is drawn every evaluation and the extent is in the cache key.

Before a commit run, in this order: `cargo fmt --all`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo test --workspace` (about 40 s;
the example-replay test is the slow one), and every replay on both backends:
`for f in examples/*/tests/*.replay; do for b in gpu cpu; do
./target/release/orla replay $f --backend $b; done; done` (about 20 s after
`cargo build --release`). Tests in one binary run in parallel: a test that
writes files takes a directory of its own (`scratch_in`, or a named scratch).

The Blender bridge is tested headless: `orla serve examples/mesh/scene.osn
--listen 127.0.0.1:0` (port on stderr) then
`/Applications/Blender.app/Contents/MacOS/Blender -b --python
bridges/blender/test_bridge.py -- --port <port> --out <dir>`. The user's own
Blender runs an MCP add-on on 127.0.0.1:9876 that executes Python sent as
`{"type":"execute","code":...,"strict_json":false}` terminated by `\0`,
answering the same way; the bridge was driven live through it (push with Live,
pull, Watch, publish, `param level`). A play session's Player and scratch
files may be left running or gone; nothing of it is in the repository.

## Next

A second bridge, to prove bridges are not Blender-specific: Godot or the
web. A web bridge is the smaller and the more telling one: a page that
opens the socket (through a small relay, since browsers speak WebSocket,
or a `--listen` that speaks it), shows the frame `read` returns, sends the
pointer and keys with `evaluate`, and lets a text area `publish` a Kernel;
the bridge in JavaScript only, outside the workspace, speaking the
protocol and nothing else, tested headless the way the Blender one is.
Acceptance: `bridges/web` drives `examples/text` and `examples/voxel`
(labels and a tool) from a browser with no Rust change beyond, at most,
a WebSocket framing in the listener, and its README says what a bridge
needs and no more. If the user prefers another item, the rest of the
ranked list is: labels on the voxel editor's bar through the text
library, caching the CPU's compiled program across frames, atomics for
the order-free scatter rules on the GPU, and `window()` in the protocol's
headless `serve` (today a client passes `window` with each `evaluate`).

## Open

Not hand-tested: the Player's real right and middle buttons and wheel,
gamepad hardware (none connected), and a live resize of a window-sized
scene in the Player (the Player was opened on the voxel editor and
presented; the resize mechanism is covered by the graph test, not by a
hand on the window's corner). The Blender bridge ran on Blender 5.2
only; Watch polls (default 0.5 s) and reads whole rows each tick. The file
watch polls every 200 ms and can observe a half-written file; the next write
is picked up. The CPU interpreter is the reference; it compiles to
word-addressed code and runs the raymarch scene at 400x260 in about 0.35 s
and a voxel frame at 960x540 in about 1.3 s on ten cores, so it is still
not interactive for those. The worker is one FIFO thread.
Audio pacing is one block behind the ring. Native manifests print element
types as anonymous structs.
