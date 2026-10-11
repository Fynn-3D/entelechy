# Typed Field Resource Architecture

**Answers:** why there is one resource, and what that binds.

Numerical resource data is a typed Field: an element schema, a logical
domain, and backend-owned storage. This document binds types, ports,
storage, execution and GPU residency inside the boundaries the two
architecture documents set. It changes by an explicit decision, never by
domain-specific exceptions in code. Code and tests establish what is
implemented; HANDOFF.md holds the next step.

## One numerical resource model

A numerical Field has a complete element schema and an explicit domain.
Records and fixed arrays compose schemas; they create no new port
transport, resource manager or execution engine. Layout is prepared from
the schema and is private to the backend; CPU columns and GPU buffers
may differ physically.

Image is an image-facing view of grid data, with explicit compatibility
spelling, sampling authority and presentation semantics. Image and Field
share resource ownership, residency, budgets, transfer and retirement. A
texture representation may implement image sampling without changing
logical resource identity or entering the IR.

Audio sample blocks and geometry attributes are numerical Fields. Sample
rate, channels, formats, vertex and index roles, and other semantic
constraints are typed metadata or capability contracts. There is no
AudioRuntime, MeshRuntime, or other resource subsystem for numerical
data.

Owned values stay values: Text, List, Option and tagged payloads keep
their ownership and complete schemas, and are neither numerical GPU
buffers nor single-element Fields. A new atomic operation may need
backend support; a new composition of supported numerical members reuses
existing layout and execution.

## Ownership and authority

- Resource identity, schema, representation validity and lifetime are
  separate from native allocation addresses. The issuing owner resolves
  handles and rejects stale, foreign or incompatible bindings. The Graph
  sees typed values and opaque handles, never wgpu objects or global
  device state.
- A resource has a CPU representation, a GPU representation, or both. An
  unchanged GPU representation is reused across nodes and frames.
  Materialization is explicit, at a CPU consumer or an external
  observation. An immutable read hides no allocation and no GPU wait. A
  successful CPU write invalidates the GPU representation; a failed write
  preserves logical contents.
- One GPU service owns byte admission, submission serialization,
  completion validation and quarantine for every numerical resource.
  Pending work retains its allocation leases and reservations. A timeout
  is not cancellation. A completed arithmetic failure discards output; a
  device failure never silently reruns the program on the CPU. No global
  cache adopts foreign Host handles.

## Computation and effects

- One validated IR describes computation independently of storage and
  devices. Within a backend, instruction lowering is shared; bindings
  supply resource access and layout. Image and Field wrappers have no
  arithmetic languages of their own.
- Execution mode is distinct from element type: once-per-invocation
  computation, domain maps and ordered transactions have different
  semantics. Errors are eager, mutation explicit, completion per node. A
  transaction is not parallelized because its data is in a Field.
- One invocation owns its provisional outputs. Output signatures,
  domains and execution status are validated before publication. A
  failed prepare or execution preserves the previous valid program and
  caller-owned inputs. A prepared plan retains its schema identity and
  immutable Kernel version.
- GPU-required preparation rejects unsupported IR. CPU fallback is a
  distinct policy, never evidence of GPU support. Performance evidence
  separates preparation, upload, dispatch and completion, and
  materialization.

## Domain capabilities

Image sampling, rasterization, audio device IO and presentation may have
specialized implementations; they reuse resource lifetimes and authority
rules. Audio scheduling respects device deadlines: a real-time callback
inherits no synchronous GPU wait and no unbounded allocation. Data being
representable implies no audio or 3D subsystem.

Synthesis, mixing, geometry processing and application rules are Kernels
wherever primitives express them. Native adapters provide device access,
not product algorithms.

## Acceptance

1. Typed numerical Fields and the Image view share one GPU resident
   lifecycle, with shared budgets, error scopes and terminal retirement,
   and one transfer and submission path.
2. A chain of numerical struct maps runs with one initial upload, no
   intermediate readback, explicit final materialization, and exact
   integer transport; Image residency, foreign-handle rejection, failed
   output cleanup and immutable input reuse hold.
3. Backends share instruction emission and invocation plumbing without
   changing authored source or wire labels; shared lifecycle helpers alone
   do not meet this.
4. A device capability extends only on this resource model: a new schema
   combination needs no other runtime or resident store, and each
   boundary has executable evidence.

Unmet obligations are HANDOFF.md's. Infrastructure is verified in Rust;
authored computation and predicates are `.ent`, composed in `.esn`,
with declarative replay.

## Where

The [resource ABI](../crates/core/abi/src/lib.rs), the
[Host that owns every Field](../crates/core/host/src/lib.rs), the
[GPU backend](../crates/backends/gpu/src/lib.rs), and the
[residency and parity tests](../crates/backends/gpu/tests/parity.rs). The
failure and authority contracts are in
[IMPLEMENTATION](../IMPLEMENTATION.md) under *Resource and residency*.
