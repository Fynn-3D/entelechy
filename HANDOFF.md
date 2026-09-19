# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits are
authorized; do not merge or push. Preserve both architectural bibles verbatim.

This implementation started on 2026-09-19 from the architecture documents alone.
The previous implementation lives outside the repository at `../bak/orla` as a
read-only reference; study its invariants, do not port it wholesale. Its typed
Field decision remains the binding direction for numerical resources when they
are introduced; the current Host exposes only images.

The runtime crates have no external dependencies; only the Player depends on
winit and softbuffer. The language covers the Phase 1 constructs plus
structured control flow: one `compute` block per Kernel, one Image output,
helper functions without recursion, Float/Int/Bool/Vec2/Vec3/Vec4 parameters,
vector intrinsics, `time()` behind `time.read`, `if` expressions, and bounded
`loop` expressions carrying one value. Structs and multi-value loops are the
next language step when a demo needs them. `.osn` scenes bind nodes to
`KernelId`s; the CLI session and the Player resolve sources from the
filesystem by polling modification stamps.

## Next

Add a GPU backend behind the same IR: emit WGSL from validated modules, run
one compute dispatch per node through wgpu, and keep the Host's ownership of
images by uploading inputs and reading outputs back per execution. Residency
(images staying on the GPU between nodes) is a later optimization once the
transfer cost dominates. Acceptance: the ray-march example renders
interactively in the Player with `--backend gpu`, and a test compares GPU and
CPU results of the same Kernels within a float tolerance, skipping cleanly
when no adapter exists.

After that: the creation protocol for external agents over stdin/stdout, as
described in README.

## Open

The watch loop polls every 200 ms and can observe a half-written file; the next
write is picked up because the stamp changes again. Exports are PPM only.
The CPU interpreter is the frame-rate ceiling: the material ball at 360x300
takes roughly 50 ms per frame and the ray-march example over a second on a
10-core machine in release. A GPU backend behind the same IR is the intended
fix, not a faster interpreter.
The example has no authored `tests/` directory yet: verification Kernels and
declarative replay, as AGENTS.md requires for demos, need a comparison primitive
that the language does not have. Add that before the second example.
