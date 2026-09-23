# Decisions

**Answers:** why Orla is this way and not another, where the answer is
not visible in the code.

Each entry is a decision that holds until it is replaced by name. It
states the choice, what it protects, and where the code enforces it.
The architecture documents set the boundaries; these are the rulings
made inside them.

## The Graph computes structure, never values

A scene document composes Kernels. The only computation it does is
structural: how large a node's output is (`domain`) and how many times a
node is placed (`times`). A value a node is given is written, a literal
or a graph parameter by name, and never worked out: `gain = n + 1;` is
refused, and so is `times = n + 1;`.

A `domain` expression reads values (a parameter, the window, a number a
Kernel wrote with `value(port)`), but only a count or a size comes out.
Its arithmetic is over counts and measures: whole numbers are exact and
fail rather than wrap, a measure must be made whole before it sizes
anything, and a boolean, a colour or a struct is not a size.

**Why.** Once a scene can compute values, it is a second language
beside `.orla`: untyped, without a trace, checked by nothing a Kernel is
checked by, run on the Host rather than on a backend, and the obvious
place to grow conditionals and loops. Every computation an agent writes
belongs where it is typed, traced, hot-reloaded and held to the same
result on every backend. The Graph owns relationships, the Kernel owns
computation.

**Where.** Node values and `times` go through the literal reader
([`scene`](../crates/scene/src/lib.rs), test
`a_scene_writes_values_and_never_computes_them`). A domain resolves
through a private `Size` in
[`graph/src/domain.rs`](../crates/graph/src/domain.rs), whose only way
out is a `Domain`.

**What would change it.** A structural need a Kernel cannot meet: a
Kernel can always compute a count and `value(port)` can size by it,
so a new domain term needs a case where that round trip is wrong, not
merely longer.

## A Kernel's work is bounded, in iterations

Every element of a Kernel may run a bounded number of loop iterations,
the Kernel's budget: 2^24 unless it declares `budget N;`. An element
that passes it fails the evaluation like an integer overflow: the
element is named, nothing commits, and the running system keeps its last
state.

**Why.** Loops take counts read at run time, so a Kernel an agent
publishes can ask for more work than a frame has. A failed update must
leave the running system alive, and a Kernel that never finishes is a
failure no validation catches. The budget counts iterations, not time,
so the same element fails on the interpreter, the JIT and the GPU, on
any machine, and a replay that passes today passes on a slower machine.

**Where.** [`Manifest::budget`](../crates/ir/src/lib.rs) and
[`Module::counts_work`](../crates/ir/src/lib.rs): a Kernel whose loops
cannot pass its budget, by their static bound, is not counted and pays
nothing. Each backend counts at the start of an iteration and reports
`ExecutionError::Exhausted` (test
`an_element_past_its_budget_fails_alike_on_every_backend`).

**What would change it.** A measure of work that is as deterministic
across backends and cheaper to keep, or a Kernel form that needs
unbounded iteration and cannot be split across evaluations.
