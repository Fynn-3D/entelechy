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

## Next

Open the creation surface to external agents: a process-level protocol over
stdin/stdout (JSON lines) on the `orla` binary that exposes the session
operations already present in `crates/cli/src/session.rs`: list Kernels and
manifests, inspect the graph, prepare a source and return diagnostics, publish
against an expected generation, evaluate, and export. Acceptance: a test in
`crates/cli/tests/` drives the binary as a client, receives structured
diagnostics for a broken Kernel, and observes that the running version
survived. Do not add product logic to the Host or the runtimes; the protocol is
an interface over existing contracts.

Then, in the Graph layer and backend-neutral: fuse chains of pointwise nodes
(those that only `sample` their inputs) into one module before execution,
verified by exact equality against the unfused CPU result. GPU residency
(images staying on the device between nodes, with explicit readback at
observation boundaries) comes after fusion, once transfer cost is what
remains.

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
Committed state is read through the Host each frame; on the GPU backend that
means the previous frame is uploaded again. Residency will make it a
ping-pong pair.
