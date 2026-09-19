# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits are
authorized; do not merge or push. Preserve both architectural bibles verbatim.

This implementation started on 2026-09-19 from the architecture documents alone.
The previous implementation lives outside the repository at `../bak/orla` as a
read-only reference; study its invariants, do not port it wholesale. Its typed
Field decision is implemented: the Host exposes one resource in three shapes.

The runtime crates have no external dependencies; the Player depends on winit
and softbuffer and the GPU backend on wgpu. The runtime holds one backend
chosen at construction; `prepare` builds backend state so `publish` stays a
pointer swap. The language covers the Phase 1 constructs plus
structured control flow: one `compute` block per Kernel, one output,
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

There is one resource now. A Field has a shape, `Rows<T>`, `Grid<T>`, or
`Volume<T>`, and an element layout; `Image` is an alias for `Grid<Vec4>`
and `type Name = Shape<T>;` declares more. Capabilities are `field.read`
and `field.write` for every shape. The Host stores words for everything
and offers pixel views; both backends transport words, so integers and
booleans are exact and the examples render identically on the CPU and the
GPU. Audio sample blocks and geometry attributes are rows of this model
plus device adapters, nothing more.

Audio exists as a device adapter in the Player: a `Rows<Float>` or
`Rows<Vec2>` graph output is played through a lock-free ring at the
device's rate (48 kHz when offered), one block per frame; the callback
never waits or allocates. The audio example is a synthesizer whose phase
crosses blocks through Graph state; its oscilloscope replay pins that.

Reduction exists as the second execution mode: a Kernel with `merge(a,
b)` maps over its first input and folds pairwise into one `Rows` element
on both backends, in the same tree; the exposure example meters a frame
and tone-maps by the result.

Geometry exists as the first native operation: `native.rasterize@1` takes
vertex and triangle rows and writes a surface grid (depth, normal, uv,
triangle id); the CPU reference and the GPU render pipeline agree on
interior pixels; the mesh example builds, projects, rasterizes, and
shades a torus with three Kernels around it.

Preparation is off the frame thread: the runtime prepares behind tickets on
one worker, the session republishes running Kernels through it, and the
evaluator runs groups unfused until their fused version arrives. `serve` and
the Player enable the worker; the one-shot tools prepare inline. Measured on
the GPU: a saved edit to the ray-march Kernel used to hold a frame for
about 70 ms and now holds none; the swap lands about 110 ms later.

## Next

Every device the architecture named has an adapter and both execution
modes the decision document names exist: map and reduce. What remains
is ordered work (prefix sums, sorting, IIR filters), which is a third
mode with real sequencing, and the smaller items: `domain` as an
expression so topology-changing Kernels (subdivision) can size their
output; the sample rate as context rather than a parameter the scene
repeats; letting the audio device drive evaluation when a scene has
sound; a reduction that keeps its intermediate levels resident across
frames instead of allocating its scratch per evaluation; and a `scan`
that reuses the reduce tree for prefix sums.

## Open

The watch loop polls every 200 ms and can observe a half-written file; the next
write is picked up because the stamp changes again. Exports are PPM only.
The CPU interpreter stays the reference and the frame-rate floor: the
ray-march example takes over a second per frame there.
Life and particles have replay tests; the other examples are time-driven
renders whose pixels a replay could still pin down at fixed times.
The worker is one thread with a FIFO queue: a burst of edits prepares in
order, and a superseded edit still occupies the queue until its turn.
Audio blocks are produced at the window's frame rate: 800 frames at 60 fps
is 48 kHz exactly, but a slow frame is an underrun and a fast one a
dropped block; a six-second run showed three underruns at startup and no
drops. The device is opened at 48 kHz when offered and otherwise at its
default rate, which shifts pitch; the rate should become context.
Native manifests print their element types as anonymous structs (`S0`);
the contract text names the fields. The handle type is still called
`ImageHandle` although it names any Field.
