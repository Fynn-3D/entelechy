# Typed Field Resource Architecture

**Answers:** why there is one resource, and what that binds.

## Decision and authority

Orla adopts the previous implementation's typed Field model for numerical
resource data: an element schema, a logical domain, and backend-owned storage.
This is the implementation decision for resource convergence, authorized by the
user. Read it before changing types, ports, storage, execution, or GPU residency.

The two original architecture documents remain unchanged and authoritative for
layer boundaries. This document records how to implement those boundaries; it is
not a feature inventory. Change this decision explicitly, not by accumulating
domain-specific exceptions in code. HANDOFF.md identifies the next executable
step; code and tests establish what is implemented.

## One numerical resource model

A numerical Field has a complete element schema and an explicit Rows/Grid domain.
Records and fixed arrays compose schemas; they do not create new port transports,
resource managers, or execution engines. Layout is prepared from the schema and
remains private to the backend. CPU columns and GPU buffers may differ physically.

Image is a typed image-facing view of grid data. Its compatibility spelling,
sampling authority, and presentation semantics remain explicit. Image and Field
share numerical resource ownership, residency, budgets, transfer, and retirement.
Future texture representations may implement image sampling efficiently without
changing logical resource identity or leaking texture objects into IR.

Audio sample blocks and geometry attributes use this numerical data foundation.
Sample rate, channels, formats, vertex/index roles, and other semantic constraints
remain typed metadata or capability contracts. Do not introduce AudioRuntime,
MeshRuntime, or another complete resource subsystem to carry numerical data.

Ordinary owned values remain values. Text, List, Option, and tagged payloads keep
their ownership and complete schemas; do not force them into numerical GPU
buffers or disguise them as single-element Fields. A new atomic operation may
require backend support. A new composition of supported numerical members must
reuse existing layout and execution mechanisms.

## Ownership and authority

Keep resource identity, complete schema, representation validity, and lifetime
separate from native allocation addresses. The issuing owner resolves handles and
rejects stale, foreign, or incompatible bindings. Graph sees typed values and
opaque handles, never wgpu objects or global device state.

A resource may have a CPU representation, a GPU representation, or both. Reuse
an unchanged GPU representation across nodes and frames. Materialize explicitly
at a CPU consumer or external observation boundary. Never hide allocation or a GPU
wait in an immutable read. A successful CPU write invalidates the corresponding
GPU representation; failed writes preserve logical contents.

The same GPU service owns byte admission, submission serialization, completion
validation, and quarantine for every numerical resource. Pending work retains
allocation leases and their reservations. A timeout is not cancellation.
Completed arithmetic failure discards output; device failure must not silently
rerun the program on CPU. Avoid a global cache that adopts foreign Host handles.

## Computation and effects

One validated IR describes computation independently of storage and devices.
Within a backend, share instruction lowering; bindings supply resource access and
layout. Image and Field wrappers must not grow separate arithmetic languages.

Execution mode is distinct from element type: once-per-invocation computation,
domain maps, and ordered transactions have different semantics. Preserve eager
errors, explicit mutation, and per-node completion during convergence. Do not
parallelize a transaction merely because its data is stored in a Field.

One invocation owns provisional outputs. Validate complete output signatures,
domains, and execution status before publication. A failed prepare or execution
must preserve the previously valid program and caller-owned inputs.
Prepared plans must retain schema identity and their immutable Kernel version.

GPU-required preparation rejects unsupported IR. CPU fallback is a distinct
policy, never evidence of native GPU support. Performance evidence separates
preparation, upload, dispatch/completion, and materialization.

## Domain capabilities

Image sampling, rasterization, audio device IO, and presentation may have
specialized implementations. They reuse resource lifetimes and authority rules.
Audio scheduling must respect device deadlines; a real-time callback must not
inherit synchronous GPU waiting or unbounded allocation. No audio or 3D subsystem
is implied to exist merely because its data can be represented.

Higher-level synthesis, mixing, geometry processing, and application rules belong
in Kernels wherever primitives can express them. Native adapters provide device
access, not product algorithms.

## Convergence order and acceptance

1. Reuse the existing GPU resident lifecycle for typed numerical Fields and the
   Image compatibility view. Remove duplicate transfer/submission machinery as
   it is superseded. Retain shared budgets, error scopes, and terminal retirement.
2. Prove a chain of numerical struct maps with one initial upload, no intermediate
   payload readback, explicit final materialization, and exact integer transport.
   Preserve existing Image residency tests, foreign-handle rejection, failed
   output cleanup, and immutable input reuse.
3. Converge backend instruction emission and invocation plumbing without changing
   authored source or wire labels. Shared lifecycle helpers alone do not establish
   this acceptance criterion.
4. Extend device capabilities only after their resource foundations reuse this
   model. New schema combinations must not require another runtime or resident
   store. Add executable evidence at each boundary.

These are acceptance obligations, not a statement that all steps are complete.
Keep unfinished boundaries in HANDOFF.md instead of inventing completion scores.
Keep infrastructure verification in Rust and authored computation/predicates in
.orla with .osn composition and declarative replay.

## Reference evidence

The old implementation is a read-only reference at
the previous implementation: `crates/orla-kernel/src/field.rs` defines schema,
domain, typed CPU columns, and GPU storage; `gpu.rs` reuses resident Fields and
explicit readback. Its native device exposure and process-wide context are not
part of this decision. Its audio tone adapter is not a general Kernel DSP model.

In this repository, start with the [resource ABI](../crates/core/abi/src/lib.rs),
the [Host that owns every Field](../crates/core/host/src/lib.rs), the
[GPU backend](../crates/backends/gpu/src/lib.rs), and the
[residency and parity tests](../crates/backends/gpu/tests/parity.rs). The failure and
authority contracts are kept in
[IMPLEMENTATION](../IMPLEMENTATION.md) under *Resource and residency*.
