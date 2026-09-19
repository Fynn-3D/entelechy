# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits are
authorized; do not merge or push. Preserve both architectural bibles verbatim.

This implementation started on 2026-09-19 from the architecture documents alone.
The previous implementation lives outside the repository at `../bak/orla` as a
read-only reference; study its invariants, do not port it wholesale. Its typed
Field decision remains the binding direction for numerical resources when they
are introduced; the current Host exposes only images.

The runtime crates have no external dependencies; the Player depends on winit
and softbuffer and the GPU backend on wgpu. The runtime holds one backend
chosen at construction; `prepare` builds backend state so `publish` stays a
pointer swap. The language covers the Phase 1 constructs plus
structured control flow: one `compute` block per Kernel, one Image output,
helper functions without recursion, Float/Int/Bool/Vec2/Vec3/Vec4 parameters,
vector intrinsics, `time()` behind `time.read`, `if` expressions, bounded
`loop` expressions carrying one value, and structs so that one value can be
a bundle. Structs stay inside the Kernel; the Typed Field decision governs
how record data will eventually cross the ABI. `.osn` scenes bind nodes to
`KernelId`s; the CLI session and the Player resolve sources from the
filesystem by polling modification stamps.

Documents are a public contract now: `.orla` libraries and `.osn` graphs
share one grammar ([docs/DOCUMENTS.md](docs/DOCUMENTS.md)); a node
references a Kernel by id or carries it inline; `use` brings in libraries.
Graphs expose named inputs, params, and outputs and are placed as nodes with
`node name: graph "path.osn" { ... }`; placement flattens the sub-graph into
the parent with prefixed node ids, so the evaluator and state keys are
unchanged.

State and input now exist: `<~ ... from ...` bindings hold Graph state with
atomic end-of-evaluation commit, and `pointer()`, `pointer_down()`, `key()`
read context behind `input.read`. `orla replay` runs an example's `.replay`
tests; the Life example is the first with a `tests/` directory.

The creation protocol exists: `orla serve` speaks JSON lines over stdin and
stdout ([docs/PROTOCOL.md](docs/PROTOCOL.md)); the end-to-end test drives
the binary as a client.

Fusion exists: pointwise chains run as one detached Kernel version, bit-
identical to the materialized chain, with pinning to observe absorbed nodes.

Residency exists: GPU images stay on the device between nodes and frames;
the Host owns them as resident slots; `Runtime::image` is the observation
boundary.

The Player presents resident outputs on the device through a blit shader;
downloads happen only on the CPU path or when the window cannot take a
frame.

The Player serves the protocol with `--serve`: requests arrive on stdin
between frames, `evaluate` answers with a window frame under the person's
input, and `read` observes the shown output.

## Next

The next capability is numerical resources beyond images, following the
Typed Field decision: a `Field` port type with an element schema and a
Rows or Grid domain, stored by the backend, so Kernels can produce and
consume point sets, audio blocks, and geometry attributes through the same
handles, budgets, residency, and commit rules images use. Start from one
demo that cannot be an image (particles as a Rows field of a struct schema)
and let it pull the primitives; do not add a second resource lifecycle.

Two smaller items first if they block a demo: prepare fused groups on a
worker thread so republishing a fused Kernel cannot hitch a frame, and let
the Player accept `--param` and `--library` like `run` does.

## Open

The watch loop polls every 200 ms and can observe a half-written file; the next
write is picked up because the stamp changes again. Exports are PPM only.
The CPU interpreter stays the reference and the frame-rate floor: the
ray-march example takes over a second per frame there. On the GPU backend
each node still uploads its inputs and reads its output back through the Host,
which bounds frame rate by transfer at large domains; residency is the answer,
after fusion.
Only Life has replay tests; the other examples are time-driven renders whose
pixels a replay could still pin down at fixed times.
Fused groups are prepared at evaluation time: on the GPU that is a shader
compile on the rendering thread, so republishing a fused Kernel can hitch
one frame. Preparing fused groups on a worker thread is the fix.
