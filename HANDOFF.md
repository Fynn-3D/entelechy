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
stays in `compute` while functions with Field parameters use `at`/`at_linear`.

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

Text rendering, the largest gap for tools: a glyph atlas as a PNG input
(baked by a Kernel with `run --save`, or imported) plus a documented row layout
for text (`Rows<Glyph>` of code, position, size) and a Kernel that draws it,
with no new primitive. Acceptance: an example (or the voxel editor's bar)
draws a label; its replay pins pixels of the text on both backends; the
atlas is an asset the Host watches. If the user prefers another item, the
ranked list from the last assessment is: CPU interpreter speed (a bytecode or
JIT path), a scatter primitive with a stated determinism rule, window size
and DPI as context, a second bridge (Godot or web) to prove bridges are not
Blender-specific.

## Open

Not hand-tested: the Player's real right and middle buttons and wheel, and
gamepad hardware (none connected). The Blender bridge ran on Blender 5.2
only; Watch polls (default 0.5 s) and reads whole rows each tick. The file
watch polls every 200 ms and can observe a half-written file; the next write
is picked up. The CPU interpreter is the reference and the frame-rate floor
(seconds per frame for ray marching at 720p). The worker is one FIFO thread.
Audio pacing is one block behind the ring. Native manifests print element
types as anonymous structs.
