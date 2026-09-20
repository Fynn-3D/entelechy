# AGENTS.md


Read in this order:

1. `AGENTS.md` — principles and engineering rules
2. `README.md` — public portal, vision, and architecture
3. `Orla Architecture Plan.md` — phased architecture roadmap
4. `Orla — Minimal Core Architecture.md` — core model and boundaries
5. `Orla Typed Field Architecture.md` — binding resource convergence decision
6. `IMPLEMENTATION.md` — code and test entry points
7. `Orla Cross-Session Handoff.md` — handoff procedure
8. `HANDOFF.md` — session context and next action
9. `Orla Git Workflow.md` — repository workflow
10. `Orla Code Rules.md` — code rules

Then inspect:

- `git status`
- recent commits
- files relevant to `HANDOFF.md -> Next`

Repository state is authoritative.
Do not rely on previous-session memory.

## Documentation maintenance

- `Orla Architecture Plan.md` and `Orla — Minimal Core Architecture.md` are
  Orla's architectural bibles. Preserve their original text verbatim. Do not
  edit, abridge, restructure, or append clarifications without an explicit user
  request to change those documents. General cleanup, consistency reviews, and
  implementation decisions do not authorize changes to these originals.
  Record subsequent interpretations and implementation decisions separately.

- README is the public vision portal; the two architecture documents define
  boundaries and roadmap acceptance criteria, not implementation status.
- IMPLEMENTATION.md links to code and executable evidence. USAGE contains a small
  set of reproducible commands. Keep schemas, versions, limits, and supported
  syntax with their implementation and tests instead of copying them into prose.
- Write documentation only for durable intent, non-obvious contracts, and useful
  entry points. Do not maintain feature inventories, test counts, scores,
  completion percentages, or chronological implementation appendices.
- Update a contract or usage example when its semantics change. Check links when
  moving files. Do not require every code change to edit a status checklist.
- HANDOFF preserves the next action and non-obvious constraints. Git preserves
  history. Dated benchmark artifacts describe experiments, not current guarantees.

## Orla

You are building **Orla**.

Orla is not an application.

Orla is a runtime for software that can evolve while it is running.

> **Host is the world.  
> Kernel is the law.  
> Graph is the structure.  
> AI is the creator.**

---

## Core

Keep the architecture soul .

```text
Host Core
↓
Capability ABI
↓
Orla IR
↓
Kernel Runtime
↓
Graph Runtime
↓
AI / Interface
```

Never collapse these boundaries for convenience.

---

## Host

The Host provides primitives.

```text
Memory
Compute
GPU
IO
Resources
Tasks
Security
```

The Host must not contain product logic.

If a feature can live in a Kernel, it should not live in the Host.

**Keep the Host boring, stable, and small.**

---

## Kernel

A Kernel is a capability.

```text
Y = K(X, P, C)
```

Where:

```text
X = Input
P = Parameters
C = Context
Y = Output
```

Prefer Kernels that are:

```text
Small
Typed
Pure
Safe
Deterministic
Composable
Hot-reloadable
```

Never give Kernels unrestricted access to the machine.

---

## ABI

The ABI is sacred.

Kernels access the Host through explicit capabilities and opaque handles.

```text
Handle          (the one handle in code: any Field, images included)
TextureHandle
BufferHandle
ResourceHandle
```

No raw pointers.

No hidden globals.

No accidental host coupling.

No capability means no access.

---

## IR

`.orla` source is not the machine.

```text
.orla
↓
AST
↓
Orla IR
↓
Backend
```

Keep Orla IR independent from CPU, GPU, WASM, Metal, CUDA, or any single backend.

The frontend describes computation.

The backend decides how it runs.

---

## Hot Reload

Hot reload is not a feature.

It is a core invariant.

```text
Compile New
↓
Validate
↓
Prepare
↓
Atomic Swap
↓
Invalidate Dependents
```

Never destroy the working Kernel before its replacement is valid.

A failed update must leave the running system alive.

---

## Graph

Kernel defines capability.

Graph defines composition.

```text
Kernel = Capability
Graph = Program
```

Nodes reference stable `KernelID`s.

They do not depend on source paths or host implementation classes.

Graph owns relationships.

Kernel owns computation.

Do not mix them.

---

## State

Prefer stateless computation.

Separate:

```text
Persistent State
Runtime State
Cache State
UI State
```

Never mix UI state with computational truth.

Make mutation explicit.

Make side effects explicit.

---

## AI

Orla is built for AI to create through Kernels and Graphs. AI Runtime and
AI Control Layer describe creative responsibilities, not a required deployment
location. Keep creation contracts open to any agent or AI-native CLI, independent
of model provider. All integrations follow the same authority and validation rules.

AI may:

```text
Generate Kernel
Generate Graph
Compile
Repair
Compose
Observe
Optimize
```

AI must not bypass the Capability ABI by injecting arbitrary native behavior.

The safe path is the powerful path.

---

## Engineering Rules

Follow `Orla Typed Field Architecture.md` for types, ports, resource storage and
execution. Numerical domains share typed Field storage and GPU residency; Image
is a compatibility view with explicit sampling/presentation semantics. Do not add
parallel resource lifecycles or execution engines for new element types, audio,
or geometry. Preserve owned-value semantics and backend-neutral ABI boundaries.

Keep runtime implementation, launchers, and runtime infrastructure tests in Rust.
Application behavior and verification predicates belong in `.orla`, composition
in `.osn`, and input replay in declarative data. Application authors must not need
new Rust test code or a native rebuild to validate their work.
Keep each demo in `examples/<demo>/`, with its project entry, `kernels/`, and
authored `tests/`. Demo-specific documentation belongs in that directory. Follow
`examples/README.md`; do not add flat examples or a redundant project manifest.
Keep the runtime crates (host, abi, ir, compiler, language, cpu, gpu, runtime,
graph, and the `orla` CLI) free of external dependencies; the interface layer
(the Player, `orla-import`, `orla-export`, `orla-jit`) is the only place that
may take them, for devices, for formats that are not worth decoding twice,
and for a code generator. `orla-jit` is also the one crate allowed `unsafe`:
it calls the code it produces, and nothing else.
Bridges to other programs live in `bridges/<program>/`, written in that
program's own language (Blender's Python), outside the Rust workspace; they
speak only the public contracts (the protocol and the Host's file formats),
and nothing in the workspace depends on them.
Do not introduce scripts or interpreter dependencies in other languages.
Data formats and documentation are not executable
language dependencies. External agents remain free to use any implementation
language when communicating through Orla's public contracts.

Keep `.orla` for Kernel source and `.osn` for declarative scene composition, with
external Kernel references or inline `.orla` bodies. JSON remains a structured
protocol representation. Source resolution belongs to the caller or file loader. Scene
documents contain authored initial state, not implicit live-state checkpoints.

Before adding complexity, ask:

**Does this belong in the Host, or can it be a Kernel?**

Before adding an abstraction, ask:

**Does this protect a real boundary?**

Before adding state, ask:

**Can this remain pure?**

Before coupling two modules, ask:

**Can they communicate through an ID, handle, IR, or capability instead?**

Before optimizing, preserve correctness and observability.

Before rewriting, understand the existing invariant.

---


##  Language

All developer-facing project communication must use clear international English.

This includes:

code comments
documentation
Git commit messages
pull request titles and descriptions
error messages intended for developers
public API documentation

Do not mix languages inside source code unless required by user-facing localization.

Keep English concise, technical, and globally understandable.

---


## Build Order

Protect this order:

```text
1. Host Core
2. Capability ABI
3. Minimal .orla Language
4. Orla IR
5. Compiler
6. Kernel Runtime
7. Atomic Hot Reload
8. Kernel Registry
9. Graph Runtime
10. Backends
11. AI Runtime / Creation Interface
12. Interfaces
```

Do not build the beautiful editor before the runtime is real.

---

## First Proof

Orla becomes real when:

```text
.orla compiles

Kernel calls Host only through ABI

Kernel hot-swaps without restart

Graph references KernelID
```

Everything else is expansion.

---

## Style

Prefer:

```text
Small modules
Explicit ownership
Stable interfaces
Immutable values
Pure functions
Clear errors
Inspectable state
Deterministic behavior
```

Avoid:

```text
God objects
Magic state
Hidden side effects
Deep inheritance
Premature frameworks
Host-specific business logic
Backend leakage
```

---

## Final Rule

Do not build features into Orla.

Build the conditions from which features can emerge.

> **Keep the world stable.  
> Keep the laws replaceable.  
> Keep composition open.  
> Let capability grow.**
