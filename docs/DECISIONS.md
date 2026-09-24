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
([`scene`](../crates/frontend/scene/src/lib.rs), test
`a_scene_writes_values_and_never_computes_them`). A domain resolves
through a private `Size` in
[`graph/src/domain.rs`](../crates/core/graph/src/domain.rs), whose only way
out is a `Domain`.

**What would change it.** A structural need a Kernel cannot meet: a
Kernel can always compute a count and `value(port)` can size by it,
so a new domain term needs a case where that round trip is wrong, not
merely longer.

## Work is bounded, in iterations, per element and per run

Every element of a Kernel may run a bounded number of loop iterations,
the Kernel's budget: 2^24 unless it declares `budget N;`. And one run of
a Kernel, all its elements together, may take a bounded number, the
Host's: four thousand million unless a session is given `--work`. An
element past its budget fails the evaluation like an integer overflow,
named; a run past the Host's fails as a run. Nothing commits, and the
running system keeps its last state.

**Why.** Loops take counts read at run time, so a Kernel an agent
publishes can ask for more work than a frame has. A failed update must
leave the running system alive, and a Kernel that never finishes is a
failure no validation catches. A bound per element is not enough: a
million elements each within theirs can still hold a device until its
driver resets it, which takes the running system with it. The Kernel
owns its budget, since it knows its loops; the Host owns the run's,
since it owns the machine, as it owns memory.

Both count iterations, not time, so what fails fails on the
interpreter, the JIT and the GPU, on any machine, and a replay that
passes today passes on a slower machine. Whether a run passes the
Host's bound is its elements' exact sum, so it is decided alike however
they are scheduled. A run that passes it and also has a failing element
may be named for either, since a run that has passed is stopped before
every element runs; it fails either way.

**Where the rule lives.** Where work is counted is written into the IR,
once: [`orla_ir::account`](../crates/core/ir/src/account.rs) follows every
loop with a `Spend` of the iterations whose body ran, ends every element
with a `Look`, and puts a `Look` first in the `until` of every loop that
can run a lot. The runtime accounts a module as it hands it to a
backend, and a backend implements what a spend and a look are, as it
implements any other operation, never where they go.

**How it stays free.** A loop adds the iterations whose body ran when it
finishes, not at each one, so an innermost loop pays one addition. An
element looks at its count when it finishes; only a loop that can run
more than 2^20 iterations by itself, its nested loops' included, also
looks at the start of each of its iterations, so an element runs at most
that far past a bound before it is stopped. The run's shared count is
touched in blocks of 1024 iterations, and what elements run short of a
block is summed where they ran (a thread, a workgroup) and handed on in
bulk. A Kernel whose elements cannot run 1024 iterations, or pass their
budget, is not counted at all. Measured against uncounted runs, the loop
scenes cost what they did.

**Where.** [`account`](../crates/core/ir/src/account.rs),
[`Manifest::budget`](../crates/core/ir/src/lib.rs),
[`Host::work`](../crates/core/host/src/lib.rs),
[`Run`](../crates/core/rules/src/lib.rs) and
[`Module::counts_work`](../crates/core/ir/src/lib.rs). Tests
`an_element_past_its_budget_fails_alike_on_every_backend` and
`a_run_past_the_hosts_work_fails_alike_on_every_backend`.

**What would change it.** A measure of work that is as deterministic
across backends and cheaper to keep, or a Kernel form that needs
unbounded iteration and cannot be split across evaluations.

## A few formats are read live; the rest are bytes or converted

The Host knows one format of its own, `.field`, and raw bytes. Every
other format a file comes in goes one of three ways:

- **Read live** by [`orla-formats`](../crates/interface/formats/src/lib.rs): PNG,
  PPM, PGM, WAV and MIDI. A scene's input names the file, and the
  session decodes it again whenever it changes.
- **Bytes to a Kernel**: `input level: Rows<Int> = bytes "level.bin";`,
  and what the bytes mean is the Kernel's business. A level, a ROM, a
  text such as Markdown.
- **Converted outside the runtime** by `orla-import` and `orla-export`,
  or by a bridge written in another program's language: glTF, MP3 and
  the other lossy sounds, and documents such as a PDF or a slide deck,
  which come in as the pictures and text they hold.

A format is read live when it decodes to a numerical Field, needs no
dependency outside the workspace, and is worth editing while a scene
runs.

**Why.** A decoder that holds a compression algorithm (inflate, a
Huffman table) is not a Kernel's work: it is serial, it would be run
again by every scene, and it carries no meaning of the application's. A
format whose meaning *is* the application's belongs in a Kernel, where
it is typed, traced and hot-reloaded. A document is not a Field at all;
bringing it past the runtime's edge would grow a subsystem for it, which
the typed-field architecture refuses. Keeping the live set closed keeps
`orla-formats` from becoming where every format lands.

**Where.** [`orla-formats`](../crates/interface/formats/src/lib.rs) takes the Host
and nothing else; `orla-import` and `orla-export` depend on it and the
Host, never on the session or a backend. `read`, `read_bytes` and
`encode` are the only ways a session turns a file into a Field and back.

**What would change it.** A format that meets all three conditions (it
joins the live set), or one the live set holds that starts to need a
dependency (it leaves for `orla-import`).

## A public contract grows; it breaks only with its version

Six things are written against by programs and sources Orla does not
see: the protocol an agent speaks, the C ABI, the capabilities a Kernel
declares, the `.field` file, what a Kernel may call and write, and what
a scene document may say. Each may grow. What was valid against it
stays valid, or its version changes.

| Contract | Held by |
| --- | --- |
| Protocol operations | the table in [the protocol module](../crates/interface/protocol/src/lib.rs), held to `OPERATIONS` and to the server answering each |
| C ABI | [`header.rs`](../crates/interface/ffi/tests/header.rs): the header's functions, version and struct layouts against the library's |
| Capabilities | [`contract.rs`](../crates/core/abi/tests/contract.rs): the set, pinned to `ABI_VERSION` |
| `.field` | [`format.rs`](../crates/core/host/tests/format.rs): every scalar and a spread of encodings pinned byte for byte, and every file in the repository still read |
| Language surface | [`surface.txt`](../crates/frontend/compiler/tests/surface.txt) at `LANGUAGE_VERSION`, held by [`surface.rs`](../crates/frontend/compiler/tests/surface.rs) |
| Scene documents | [`surface/`](../crates/frontend/scene/tests/surface/): documents writing every form a scene has, each beside the Graph it lowers to at `LANGUAGE_VERSION`, held by [`surface.rs`](../crates/frontend/scene/tests/surface.rs) |

**Why.** A contract that changes by accident breaks someone who is not
in the room: a Blender scene saved last month, a Kernel an agent wrote
yesterday, a C program built against last week's header. A test that
fails on any difference makes every change a decision, and the version
says which kind it was.

**Where.** The tests above. The language's surface and the scene
documents are the ones recorded as data. A new document in the scene
corpus is recorded with `ORLA_BLESS=1`; one that lowers differently is
refused until `LANGUAGE_VERSION` is raised. For the language: a new
builtin or signature, or a capability no longer needed, is recorded
with `ORLA_BLESS=1`; dropping or tightening anything a source could
write is refused until `LANGUAGE_VERSION` is raised.

**What would change it.** A contract that has no outside reader after
all, which then need not be held; or a second version of one living
beside the first, which would need both held.

## The rasterizer is a package; the native one is its fast path

A scene places `packages/raster.orlapkg` to turn triangles into a
surface, and binds `native.rasterize@1` in its place, one node for
another, where it wants speed. Both keep one rule: a pixel is covered
where its centre is inside the triangle's box and on or inside all three
edges, holds the nearest fragment in `[0, 1)`, the earlier triangle on a
tie, with attributes interpolated by `1 / w`.

**Why.** Written as Kernels, rasterization is what the rest of Orla is:
typed, traced, hot-reloaded, and the same on every backend, edges
included, because every backend runs the same IR. A native rasterizer on
a GPU is the device's, whose coverage at an edge and whose precision are
its own, so the three backends drew different surfaces; and a
rasterizer in Rust is a feature an author cannot change. The package is
slower — a frame of the culled example's world is 18 ms on the GPU
against 11 with the device's rasterizer, and on the JIT the Rust fast
path draws in 22 ms what the package draws in about 90 — so the fast
path stays, as an acceleration of the package rather than a second
definition of it.

**Where.** The package and its presets; every example places it. On the
interpreter and the JIT the fast path is `orla_rules::raster`, held bit
for bit to the rule by a test that writes the rule plainly; on the GPU it
is the device's rasterizer, which the documents call approximate.
`examples/mesh/tests/edges.replay` pins the package's surface at edge
pixels on all three backends, and `native.replay` the fast path's
inside them.

**What would change it.** A GPU fast path exact to the rule (a compute
rasterizer as fast as the device's, or integer coverage the device
honours), which would make the fast path a pure acceleration on every
backend; or a scene that needs the fast path chosen for it, which would
need the Graph to know that a placed graph and a native operation are
one computation.
