# Orla Handoff

## Context

Conversation is Chinese; source, docs and commits are English. Local commits are
authorized; do not merge or push. Preserve both architectural bibles verbatim.
Follow the Typed Field architecture. Old Orla is a read-only reference.

The user wants actual Blender-style Object/Edit Mode: vertex, edge and face
operations, extrusion, subdivision, loop cuts, Fill, inset, bevel and Knife.
That broader request is not complete. Start with
[the modeler entry guide](examples/mesh-modeler/README.md) and
[the topology contract](examples/mesh-modeler/docs/topology.md).
The old temporary editor is at
`/tmp/orla-mesh-editor-v20-history-scatter`; its drivers are not dependencies.
Record partial updates already exist; do not implement them again.

The user asked to consolidate only this demo. Its sole application entry is
`examples/mesh-modeler/project.osn`; fixed-grid stress fixtures live under
`benchmarks/`. Extend the existing mesh/render/ui Kernels and shared overlay
library. The user subsequently explicitly requested porting the existing Mesh Lab
editor, including its interactions, rather than rebuilding those behaviors.
Use its sources for this migration; do not mix stress fixtures into the application.
Test scenes and verifier Kernels belong under `tests/fixtures/`; replay JSON stays
in `tests/`. Other demos were explicitly excluded from this cleanup.

## Next

Continue the direct Mesh Lab migration described in
[the port contract](examples/mesh-modeler/docs/mesh-lab-port.md). The sole main
entry now connects modal G/R/S, numeric input, baseline restoration and a bounded
history ring. The session replay checks their interaction with discrete topology
operations. Continue migrating the original operator and viewport code, preserving
confirmation/cancellation and failure retention, instead of redesigning them.
Then migrate retained geometry and GPU submission behavior while preserving
owned-value semantics and checked failure attribution. Do not blindly remove
per-node completion checks to reproduce old renderer-only timings.
Use the [topology experiment](docs/benchmarks/MESH_TOPOLOGY_2026-09-14.md) as the
baseline, not the fixed-grid timings. The current complete viewport is not yet
Blender-like in responsiveness.

The bounded reduction tree covers the configured polygon extent. Picking and
neighbor admission are parallel face maps; they are not persistent adjacency.
Shared adjacency and deterministic sparse worklists remain necessary for region
editing and ring cuts. Study the old adjacency/split/triangulation Kernels before
selecting general Field primitives. Do not raise serial loop limits to scale.

Keep the topology and million-grid workloads distinct. The topology scene
uses reserved capacity and bounded per-face work; the earlier million-scale timings
describe fixed connectivity. Restoring a retained polygon is not general Fill.
The renderer's convex fan is not general tessellation. Modal preview/cancel,
region editing, general Fill, bevel and Knife need their own complete semantics.

All group transitions read old committed state. Editor gestures and topology
commands have explicit one-frame consumption latency; render sees the new complete
group. Geometry and metadata commit together. Do not mix UI selection with
computational truth or claim the topology revision is undo history.

Numerical Fields and rasterization reuse the existing service, opaque handles,
budgets and checked completion. Adapter binding capacity is negotiated; preparation
still rejects device limits. Native maps currently return one resource; record
Fields carry compound draw data. Do not introduce MeshHandle, a second GPU device,
or another execution engine. Uniform live seeds and bounded protocol snapshots
retain different admission rules.

Run Cargo commands serially. Benchmark release GPU work without concurrent builds,
tests or Player processes. Timings exclude window composition unless explicitly
measured. Keep per-node failures visible and preserve last-good state and leases.
