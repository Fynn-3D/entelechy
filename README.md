# Orla

**An open runtime for AI-created applications and games.**

Orla's vision is to let any AI agent create, run, and continuously evolve
interactive applications and games through Kernels and Graphs.

It provides a stable computational world in which software behavior can be
defined, composed, inspected, and replaced while the program is running.
Agents bring intent, reasoning, and creativity. Orla provides the execution
rules, resource boundaries, and language through which that creativity becomes
working software.

> **Host is the world.**
> **Kernel is the law.**
> **Graph is the structure.**
> **AI is the creator.**

## The vision

A person should be able to work with an agent of their choice to build an
interactive tool, invent a visual effect, compose an application, or develop a
game. The agent should be able to inspect what exists, introduce new capabilities,
connect them, observe the result, and refine the running program.

Orla aims to make this possible through a common creation surface. An agent's
vendor, model, planning strategy, and command-line environment should not determine
which parts of the runtime it can use. Access follows explicit capabilities and
protocol contracts.

The long-term ambition is broad: a runtime on which many kinds of applications
and games can be created. Each expansion must earn its place through working
examples and a clear architectural boundary. The current implementation proves
dynamic computation and live composition; general application and game creation
is the direction of development.

## An open creation surface

Orla is built for AI to create software through Kernels and Graphs. AI Runtime
and AI Control Layer describe creative responsibilities: understanding intent,
generating capabilities, composing programs, observing results, and refining them.
Their integration or deployment location does not define the architecture.

Any agent or AI-native CLI should be able to use the same creation contracts.
Orla remains independent of model providers; authority, validation, and publication
rules apply regardless of how an agent connects.

The intended creation loop is:

```text
Human intent
    ↓
Any AI agent / AI-native CLI
    ↓
Open creation protocol
    ↓
Inspect → Propose → Validate → Preview → Publish → Observe
              ↑                                    │
              └────────── Repair / Refine ──────────┘
    ↓
Kernel + Graph Runtime
    ↓
Live application or game
```

This protocol should let clients discover capabilities and signatures, inspect
Graph structure and versions, submit source and composition changes, receive
structured diagnostics, preview candidates, and publish against an expected
generation. Different agents should be able to use the same operations.

Transport is an implementation choice; the durable contract is the creation
workflow and its authority, ownership, and publication semantics.

## The architecture

| Layer | Responsibility |
| --- | --- |
| Host Core | Stable primitives for memory, compute, GPU, IO, resources, tasks, and security |
| Capability ABI | Explicit authority and opaque handles through which Kernels access primitives |
| Orla IR | Typed computation independent of the execution backend |
| Kernel Runtime | Preparation, execution, immutable versions, and safe replacement |
| Graph Runtime | Composition, typed connections, and relationships between capabilities |
| Interfaces | Agent protocols, CLI tools, players, editors, and other runtime clients |

These responsibilities describe the architecture. The available primitive set
grows incrementally; not every domain in the table is exposed to Kernels today.

### Host provides the world

The Host supplies primitives with explicit ownership and bounded resources.
Application rules, visual algorithms, and game behavior belong in replaceable
programs wherever the available primitives can express them.

New Host functionality must protect a real hardware or authority boundary.
Higher-level capabilities should grow through Kernels and composition.

### Kernels define capabilities

A Kernel describes computation:

```text
Y = K(X, P, C)

X = Input     P = Parameters     C = Context     Y = Output
```

Prefer small, typed, pure, deterministic, composable Kernels. Make mutation and
side effects explicit when they become necessary. Resource access uses declared
and granted capabilities with opaque handles. Generated source receives no raw
pointer or unrestricted machine access through the Kernel interface.

The current Rust backends and embedding applications remain trusted. The
capability boundary constrains Orla programs; it is not an operating-system
sandbox for arbitrary native extensions.

### Graphs define programs

Graph nodes reference stable `KernelID`s. A Kernel owns its computation; the
Graph owns connections and composition. Source paths and interface state remain
outside those relationships.

Changing a connection or binding can create new behavior without changing the
underlying Kernel. New Kernels extend what can be composed. Together, these are
the building blocks from which application and game behavior should emerge.

### IR preserves freedom of execution

```text
.orla source → AST → Validated Orla IR → Backend → Execution
```

The frontend describes computation. Backends decide how it executes. CPU and GPU
implementations share this boundary; future backends should preserve it rather
than impose their own machine model on the language or Graph.

### Change preserves the running program

Compile, validate, and prepare a replacement before publishing it. Publication
must be atomic for the affected program. Reject stale or foreign candidates.
Keep existing evaluations pinned to their immutable versions.

A failed update leaves the active program intact. A later execution failure is a
separate event: it does not automatically reverse a committed revision. Interfaces
can retain the last successful result while exposing the failure for repair.

Persistent state, runtime state, cache state, and UI state must have distinct
owners. Cross-frame state and migration require equally explicit contracts.

## Building toward the vision

Two directions guide development: make creation accessible to AI agents,
and make more application behavior expressible through Kernels and Graphs.

The first requires an open, versioned creation protocol with structured feedback
and explicit publication semantics. The second requires composable primitives
for resources, input, state, rendering, and audio, proved through small working
programs before broader applications.

Use the [implementation evidence index](IMPLEMENTATION.md) to inspect code and
executable proofs. The [usage guide](docs/USAGE.md) provides commands and examples.
The roadmaps describe acceptance criteria, not a promise of universal support.

## Design references

- [Typed Field architecture](Orla%20Typed%20Field%20Architecture.md): resource
  convergence, domain extension boundaries, and cross-session acceptance rules.

- [Architecture plan](Orla%20Architecture%20Plan.md): the broader system roadmap.
- [Minimal core architecture](Orla%20%E2%80%94%20Minimal%20Core%20Architecture.md):
  the computational model and layer boundaries.
- [Engineering principles](AGENTS.md): rules for preserving those boundaries.
- [Document format](docs/DOCUMENTS.md): `.orla` libraries and `.osn` graphs.
- [Implementation evidence](IMPLEMENTATION.md): code and test entry points.
- [Usage guide](docs/USAGE.md): setup, examples, and technical reference.
- [Current handoff](HANDOFF.md): session context and the next action.

## The guiding questions

Does this change let more agents work reliably through Orla's public interfaces?

Does it let more application behavior be expressed through Kernels and Graphs
without adding product logic to the Host?

> **Keep the world stable.**
> **Keep the laws replaceable.**
> **Keep composition open.**
> **Let capability grow.**
