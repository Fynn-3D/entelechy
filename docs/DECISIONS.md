# Decisions

**Answers:** why Orla is this way and not another, where the answer is
not visible in the code.

Each entry is a decision that holds until it is replaced by name. It
states the choice, what it protects, and where the code enforces it.
The architecture documents set the boundaries; these are the rulings
made inside them.

## The Graph computes structure, never values

A scene document composes Kernels. Its only computation is structural:
how large a node's output is (`domain`) and how many times a node is
placed (`times`). A node's value is written, a literal or a graph
parameter by name: `gain = n + 1;` and `times = n + 1;` are refused.

A `domain` expression reads values (a parameter, the window, a number a
Kernel wrote with `value(port)`) and yields only a count or a size. Its
arithmetic is over counts and measures: whole numbers are exact and fail
rather than wrap, a measure is made whole before it sizes anything, and
a boolean, a color or a struct is not a size.

**Why.** A scene that computes values is a second language beside
`.orla`: untyped, untraced, unchecked, run on the Host rather than a
backend. Every computation belongs where it is typed, traced,
hot-reloaded and held to one result on every backend; a Kernel computes
any count and `value(port)` sizes by it.

**Where.** Node values and `times` go through the literal reader
([`scene`](../crates/frontend/scene/src/lib.rs), test
`a_scene_writes_values_and_never_computes_them`). A domain resolves
through a private `Size` in
[`graph/src/domain.rs`](../crates/core/graph/src/domain.rs), whose only way
out is a `Domain`.

## Work is bounded, in iterations, per element and per run

Each element of a Kernel runs at most its budget of loop iterations:
2^24 unless it declares `budget N;`. One run of a Kernel, all elements
together, runs at most the Host's bound: four thousand million unless a
session is given `--work`. An element past its budget fails the
evaluation, named, like an integer overflow; a run past the Host's bound
fails as a run. Nothing commits; the running system keeps its last state.

Both bounds count iterations, never time. A run passes the Host's bound
by its elements' exact sum, whatever the schedule; a run that passes it
and has a failing element may be named for either. Both bounds are a
node's own: a Kernel whose work is counted is never fused.

Where work is counted is written into the IR once:
[`orla_ir::account`](../crates/core/ir/src/account.rs) follows every loop
with a `Spend` of the iterations whose body ran, ends every element with
a `Look`, and puts a `Look` first in the `until` of every loop that can
run more than 2^20 iterations itself. The runtime accounts a module as
it hands it to a backend; a backend implements what a spend and a look
are, never where they go. The run's shared count is touched in blocks of
1024 iterations, partial blocks summed where they ran (a thread, a
workgroup). A Kernel whose elements cannot run 1024 iterations, or pass
their budget, is not counted.

**Why.** A failed update leaves the running system alive, and a Kernel
that never finishes is a failure no validation catches; a bound per
element alone still lets a million elements hold a device until its
driver resets. The Kernel owns its budget because it knows its loops;
the Host owns the run's because it owns the machine. Counting iterations
makes a failure the same on every backend and machine.

**Where.** [`account`](../crates/core/ir/src/account.rs),
[`Manifest::budget`](../crates/core/ir/src/lib.rs),
[`Host::work`](../crates/core/host/src/lib.rs),
[`Run`](../crates/core/rules/src/lib.rs) and
[`Module::counts_work`](../crates/core/ir/src/lib.rs). Tests
`an_element_past_its_budget_fails_alike_on_every_backend` and
`a_run_past_the_hosts_work_fails_alike_on_every_backend`.

## What a run holds in Host memory counts against the Host's budget

A scatter's pairs (the one each element returns and every one it
`emit`s) are held in Host memory by the interpreter and the JIT, against
the room the Host's budget has left when the run begins
([`Scope::room`](../crates/core/abi/src/lib.rs)), as a Field is: the
returned pairs before the run starts, the emitted ones as they come, the
JIT's sorted copies before it sorts them. Past the room the run fails as
`OverRoom`, named for its room; nothing commits. A device that lands each
pair where it is emitted holds no list, so which backend a run fits on
is the machine's property, as for any allocation; what a run computes is
not. A run past the room with a failing element may be named for either.

**Why.** The Host owns memory as it owns work. An element that emits in
a loop holds as much as its work bound lets it, and that bound is on
iterations, not memory; a published Kernel must not take the machine's
memory by asking.

**Where.** [`Scope::room`](../crates/core/abi/src/lib.rs); `OverRoom` in
[`orla_rules`](../crates/core/rules/src/lib.rs), held by the JIT's
[parity tests](../crates/backends/jit/tests/parity.rs).

## A few formats are read live; the rest are bytes or converted

The Host has one format of its own, `.field`, and raw bytes. Every other
format goes one of three ways:

- **Read live** by [`orla-formats`](../crates/interface/formats/src/lib.rs): PNG,
  PPM, PGM, WAV and MIDI. A scene's input names the file; the session
  decodes it again whenever it changes.
- **Bytes to a Kernel**: `input level: Rows<Int> = bytes "level.bin";`.
  What the bytes mean (a level, a ROM, Markdown) is the Kernel's.
- **Converted outside the runtime** by `orla-import` and `orla-export`, or
  by a bridge: glTF, MP3 and other lossy sounds, SVG as the segments and
  shapes a Kernel fills, TrueType and OpenType fonts as the glyphs'
  outlines in the same segments, and documents such as a PDF or a slide deck, as
  the pictures and text they hold.

A format is read live when it decodes to a numerical Field, needs no
dependency outside the workspace, and is worth editing while a scene runs.

**Why.** A compression decoder is serial, carries no application meaning,
and would run again in every scene, so it is not a Kernel's work; a
format whose meaning is the application's belongs in a Kernel. A document
is not a Field, and the typed-field architecture refuses a subsystem for
it. A closed live set keeps `orla-formats` from collecting every format.

**Where.** [`orla-formats`](../crates/interface/formats/src/lib.rs) takes the Host
and nothing else; `orla-import` and `orla-export` depend on it and the
Host, never on the session or a backend. `read`, `read_bytes` and
`encode` are the only ways a session turns a file into a Field and back.

## A public contract grows; it breaks only with its version

Six things are written against by programs and sources Orla does not
see: the protocol an agent speaks, the C ABI, the capabilities a Kernel
declares, the `.field` file, what a Kernel may call and write, and what a
scene document may say. Each may grow. What was valid against it stays
valid, or its version changes.

| Contract | Held by |
| --- | --- |
| Protocol operations | the table in [the protocol module](../crates/interface/protocol/src/lib.rs), held to `OPERATIONS` and to the server answering each |
| C ABI | [`header.rs`](../crates/interface/ffi/tests/header.rs): the header's functions, version and struct layouts against the library's |
| Capabilities | [`contract.rs`](../crates/core/abi/tests/contract.rs): the set, pinned to `ABI_VERSION` |
| `.field` | [`field_file.rs`](../crates/interface/formats/tests/field_file.rs): every scalar and a spread of encodings pinned byte for byte, and every file in the repository still read |
| Language surface | [`surface.txt`](../crates/frontend/compiler/tests/surface.txt) at `LANGUAGE_VERSION`, held by [`surface.rs`](../crates/frontend/compiler/tests/surface.rs) |
| Scene documents | [`surface/`](../crates/frontend/scene/tests/surface/): documents writing every form a scene has, each beside the Graph it lowers to at `LANGUAGE_VERSION`, held by [`surface.rs`](../crates/frontend/scene/tests/surface.rs) |

**Why.** A contract changed by accident breaks a reader who is not
present: a saved scene, a generated Kernel, a program built against the
header. A test that fails on any difference makes every change a
decision, and the version says which kind.

**Where.** The tests above. The language surface and the scene documents
are recorded as data. A new scene document is recorded with
`ORLA_BLESS=1`; one that lowers differently is refused until
`LANGUAGE_VERSION` is raised. A new builtin or signature, or a
capability dropped from a requirement, is recorded with `ORLA_BLESS=1`;
dropping or tightening anything a source could write is refused until
`LANGUAGE_VERSION` is raised.

## The rasterizer is a package; the native one is its fast path

A scene places `packages/raster.orlapkg` to turn triangles into a
surface (`raster.osn` or `window.osn`; `cutout.osn` or
`cutout_window.osn` where a mask's alpha cuts fragments away) and binds
`native.rasterize@1` or `@2` in its place, one node for another, where it
wants speed. Both keep one rule: a pixel is covered where its centre is
inside the triangle's box and on or inside all three edges, holds the
nearest fragment in `[0, 1)`, the earlier triangle on a tie, with
attributes interpolated by `1 / w`; with a cutout, a fragment whose alpha
in the mask at its own uv is below the cutout is not drawn. The `tiled`
presets cut a box larger than a tile across a pool of invocations, for
scenes of large triangles; the surface is the same to the bit where
every Kernel rounds alike, as on the interpreter and the JIT. A
triangle with two corners at one point is skipped by that test alone.
Each triangle's box is found once, and every step reads it.

**Why.** As Kernels, rasterization is typed, traced, hot-reloaded, the
same on every backend, and changeable by an author. The native path is
an acceleration of the package, not a second definition: on the
interpreter and the JIT it is exact to the rule, on a GPU it is the
device's rasterizer, approximate at edges. The Graph does not know that
a placed graph and a native operation are one computation, so the scene
chooses. A device may compute one expression to different last bits in
two Kernels, as its compiler fuses a multiply into an add where it
chooses, so a decision two Kernels must share is made in one: were a
box found twice, the tiles counted from one would be cut from another.
`orla replay --round-apart` finds one made twice, by rounding some
Kernels as another device might.

**Where.** The package and its presets; every example places it.
[`ten_million.osn`](../examples/mesh/tests/ten_million.osn) is the scale
scene. On the interpreter and the JIT the fast path is
`orla_rules::raster`, held bit for bit to the rule by a test that writes
the rule plainly. `examples/mesh/tests/edges.replay` pins the package's
surface at edge pixels on all three backends and `native.replay` the fast
path's inside them; `degenerate.replay` holds that a triangle with two
corners at one point draws nothing and the tiled presets draw what the
one-pass ones do, on all three; `cutout.replay` holds the package's cutout to
`native.rasterize@2` on all three, and a session test holds them equal to
the bit on the interpreter and the JIT.

## Contacts are solved all at once, each against its own share of a body

The physics package steps rigid bodies with every contact manifold
solved in one pass: each solves against a copy of its two bodies whose
mass is split among the manifolds that push them, and the pushes on a
body are summed. A step runs substeps of gravity, the held impulses
pushed again, a solve, a move and a solve without pushing apart; each
manifold starts from the impulses the same pair held in the last step.
A contact is a spring of the bodies' own masses, so how far a pile
sinks into itself does not depend on how many manifolds share a body.

**Why.** Solving contacts one after another, as a sequential solver
does, needs an order that a pass over every manifold does not have;
colouring them into independent sets costs a pass a colour, each a
dispatch. Solved at once with split masses, a pass converges where the
one-by-one order would, its answer is independent of order (a float
sum of pushes is folded in source order), and the same on every
backend. Its cost is convergence: a pile many bodies deep that lands at
once needs more passes, which is why a step takes eight substeps, and
carrying the impulses from step to step is what lets a resting stack
stand still.

**Where.** [`physics.orla`](../packages/physics.orlapkg/kernels/physics.orla)
and its [step](../packages/physics.orlapkg/graphs/step.osn);
[`rest.replay`](../examples/physics/tests/rest.replay) holds a column
of boxes still for seconds, [`motion.replay`](../examples/physics/tests/motion.replay)
a bounce, friction on a slope and a kinematic platform to their
closed forms, and [`many.replay`](../examples/physics/tests/many.replay)
the grid's pairs to every pair asked, on all three backends.

## A small run goes to the processor; a device is for many elements

When the main backend is a device and the JIT runs beside it, each run of
a Kernel goes to the one its size suits: to the JIT when its elements
times its Kernel's typical work (`Validated::typical_work`, an estimate
of operations per element from the IR) are at most `--small`, 400,000 by
default; to the device otherwise. A Kernel the device declines always
runs on the JIT. A small run that reads a Field the device made in this
evaluation stays on the device, since bringing it back would wait for
the device's work before it; a Field made before, state or an input,
comes back once. A Field uploaded from memory keeps its words in memory
beside the device's copy, as a Field is a value, so neither side sends
it back again. The report says where each run went and why; `--small 0`
keeps every run on the device.

**Why.** A device's dispatch costs about the same whatever it does:
80 microseconds on the machine it was measured on, where a chain of two
hundred small runs takes 15.6 ms on the device and 0.73 ms on the JIT,
and a step of sixty-seven rigid bodies 10.6 ms against 0.57 ms. Submitting
them together, or in one pass, costs the same: the time is the dependent
dispatches themselves. A device earns its cost over many elements, so
where a run goes follows its size, not whether it draws or computes. The
choice reads the run's elements and the Kernel, never a clock, so a
replay places every run the same way each time. A run may round its
last bits differently on the two, so a decision two Kernels share is
made in one (see *The rasterizer is a package*).

**Where.** `Runtime::execute` in [`orla-kernel`](../crates/core/kernel/src/lib.rs),
the estimate in [`orla-ir`](../crates/core/ir/src/lib.rs), the kept
words in [`orla-host`](../crates/core/host/src/lib.rs);
[`routing.rs`](../crates/interface/session/tests/routing.rs) holds every
example's outputs to the JIT's alone with small runs placed off a main
backend, and the verify recipe replays every example on the device with
`--small 0` as well.

## A Kernel's effects are data the Host carries out after a commit

A Kernel never writes a file. A scene declares that an output is also
written to a file, `output kept <- state.result to "saves/state.field";`,
and the session writes it once an evaluation that changed it has
committed: whole (written beside and renamed over), into a directory
allowed with `--allow file.write=<dir>`, and nowhere else. An input
reads a file and reads it again when it changes; one that may be absent
falls back on a source (`input saved: … = "saves/state.field" else
fresh.result;`). When and what to save is the Kernels'.

The session decides every write: `save` and `export` check its allowance
whoever calls them and write through its store, and a file named on a
command line (`--save`, `--out`) is allowed alone, not its directory. A
server takes its allowances from the command line that started it.

**Why.** An effect inside a Kernel breaks what a Kernel is: the same on
every backend (a device has no files), committing nothing when it fails,
and replaying alike. As data, the effect is a typed, traced Field, and
the Host carries it out after the commit, where nothing can undo it. The
allowance is given by the person running the scene, never claimed by a
Kernel or an agent.

**Where.** [`writes.rs`](../crates/interface/session/src/writes.rs) in the
session; `to` and `else` in the document
([parser](../crates/frontend/language/src/parser.rs),
[lowering](../crates/frontend/scene/src/lib.rs)); a fallback in use
substitutes for its input when the evaluation starts
([evaluator](../crates/core/graph/src/evaluate/mod.rs)). Held by
[the session's write tests](../crates/interface/session/tests/writes.rs),
[the lowering's](../crates/frontend/scene/src/lib.rs) and
[`examples/save`](../examples/save/README.md).

## What a served scene reaches is what the person running it gave

A session an agent drives (`orla serve`, the Player's `--serve`, a
program's `orla_request`) holds the scene to what the person who started
it allowed; nothing over the wire widens it.

- Whether reads are held is decided when the session opens, before its
  scene is first read. Readable are the directories of the scene as it
  first loads (the root, the graphs it places, what it uses, the files its
  inputs read), the directories packages are found in, and `--allow
  file.read`, judged where a path lands after its links. A program allows more with `orla_allow`.
- An agent's `save` and a scene's `to` land only where `--allow
  file.write` says.
- A page opens the socket only from this machine or from an origin
  `--allow-origin` names; a page from a file says `null`, as a sandboxed
  frame on any site does, so it is named too; an HTTP request that is not a WebSocket
  upgrade is answered and closed.
- A package is a published capability, read by anyone as `use` reads
  it.
- A Kernel's `requires` must be among the runtime's grants (all, unless
  `--grant` names fewer) for it to be prepared; it runs with what it
  requires of those.
- What an agent sends is bounded before it costs anything: a domain is
  charged against the Host's budget before allocation, on the frame or
  off it; an `[each]` and a `times` before they are written out; a socket
  request before it is held; a rejected scene never reaches the disk.

A scene a person runs themselves (`run`, `render`, `replay`) reads what
they chose to run and writes only into the directories they allowed.

**Why.** A file an agent can read it can send away, and any page the
person visits can reach a socket on their machine. A scene the person
runs has no channel back out but its allowed writes.

**Where.** The session's
[inputs](../crates/interface/session/src/inputs.rs) and
[writes](../crates/interface/session/src/writes.rs); the protocol's
[transport](../crates/interface/protocol/src/transport.rs) and
[WebSocket](../crates/interface/protocol/src/websocket.rs); the Host's
[budget](../crates/core/host/src/lib.rs). Held by the protocol's tests
([`protocol.rs`](../crates/interface/orla/tests/protocol.rs)): saves,
reads, origins and compose; and by the C ABI's
([`abi.rs`](../crates/interface/ffi/tests/abi.rs)) for reads through
`orla_request`.

## Float `min` and `max` have one rule, and it is not the machine's

`max(a, b)` of floats is IEEE 754-2019 maximumNumber: a NaN gives way to
the other operand, and of two equal values the one without a sign, so
`max(-0.0, 0.0)` is `0.0` in either order; `min` is the same rule the
other way, `min(-0.0, 0.0)` is `-0.0`; `clamp` is `max` then `min`, and
`smoothstep` clamps so.

**Why.** A machine's `max` leaves the zero's sign and NaN to the
hardware, so one program gave `inf` and `-inf` on two backends. A
semantics a backend may choose is not one.

**Where.** The interpreter writes the rule out (`orla_rules::max_f32` and
its siblings in [`orla_rules`](../crates/core/rules/src/lib.rs)), the JIT
emits it, the GPU calls `max_ref` in the shader's
[preamble](../crates/backends/gpu/src/wgsl/prelude.rs). Held by the JIT's
random programs ([`fuzz.rs`](../crates/backends/jit/tests/fuzz.rs)).

## An integer sum is its whole total

A scatter's or a prefix's `sum` of integers is each element's whole
total: what it held and every value added to it, as mathematics adds
them. It fails only when that total does not fit the scalar, naming the
first element whose total does not; a total that passes the end of the
range and comes back is the total it comes back to. The interpreter and
the JIT keep each total wide (`orla_rules::Sums`), counted against the
run's room. The GPU adds a word atomically and carries into a high word
beside it, checks once every value is in, and, where a float is summed
beside the integer in element order, carries the high word as it folds.

**Why.** A sum's inputs, not the order a machine adds them, decide it, so
every backend answers alike, to the element that fails.

**Where.** `Sums` in [`orla_rules`](../crates/core/rules/src/lib.rs); the
GPU's [emitter](../crates/backends/gpu/src/wgsl/emitter.rs).

## Reality changes atomically; history records why

A session's position is `(revision, frame)`: the revision moves when the
world changes other than by running (a Kernel published or retired, the
scene read again, a param set, state written or reset); the frame counts
the evaluations committed within it. Every change is recorded with its
origin (its document, the protocol and its channel, or the program), and
Kernel sources are kept once each by digest, so an old generation can be
read and published again; refused attempts are recorded apart, with why,
and move nothing. A snapshot takes the world whole, committed state only,
with the sources of its laws, and is kept by whoever asked, or written to
a file where writes are allowed.

**Why.** An agent changing a live world needs to know how it came to be,
and what was refused. The frame is its own coordinate so that
discontinuous changes are not buried under per-frame state. Only
committed state is taken because the cache may hold what a failed
evaluation never committed. The session keeps changes, bounded, never
worlds: the past is data its holder keeps.

**Where.** [`history.rs`](../crates/interface/session/src/history.rs) in
the session; `history`, `kernel` with `generation` and `snapshot` in the
protocol. Held by the protocol's tests and the history's own.

## A live world is changed through three doors, each at its own grain

- **Structure**, a statement at a time: `compose` edits the scene text
  where a node, a node's member, an output or a declaration is written,
  and only there, or writes it whole. The scene is then read again as an
  edit on disk is; a text whose graph cannot run a frame (a missing
  source, a cycle) is refused before it replaces the running one.
- **State**, a committed port at a time: `write` replaces one, `reset`
  with a `node` and `port` restarts one from its initial source, `reset`
  alone restarts all, `restore` installs a world. A run off the frame
  that began from forgotten state, or from what was computed from it, is
  let go; runs that read none of it go on.
- **Parameters**, the ones the document declares (`param` in the scene).
  A node's own value is structure and changes with the document.

Nothing else reaches in.

**Why.** Each door is where the thing it changes is kept whole. A node
value set from outside is structure the document does not say, lost at
the next read and invisible to `document` and history. The document is
also a person's: an agent names what it changes and every other byte
stays. A port is the smallest part of state that restarts without
inventing a state no evaluation committed.

**Where.** [`edit`](../crates/frontend/scene/src/edit.rs) in the scene,
over the language's `outline`, and `compose` in the protocol;
`Graph::order` where the session lowers a scene. `Evaluator::reset_port`
and `reset_state` in the
[graph's state](../crates/core/graph/src/evaluate/state.rs);
`Session::reset_port`, recorded in history as a reset with its port;
`reset` in the protocol and `reset node.port` in a replay. Held by the
graph's tests (one port restarts, the others keep theirs; a reset lets
go of exactly the runs that began from what it forgot), the scene's edit
tests (every byte an edit does not name is kept), the protocol's (a
node's own value refused by `param`, changed by an edit and kept by the
document; a declared param reaching its node), and the voxel replay.

## A world is restored only when the whole world can live again

`restore` installs a snapshot's state in every state port at once, ports
it does not name restarting from their initial sources. With `laws:
"snapshot"` it first publishes the snapshot's Kernels back, all together;
with `structure: "snapshot"` it first writes back the snapshot's scene
document (old sources and documents are kept by digest). `laws` has no
default. It then proves the world by one frame, migrating state as any
frame does, and commits nothing: on success the world is the snapshot's
and the next frame follows it; on failure state and laws are put back.

**Why.** Every change is prepared, validated and proved before it is
real; a restore is the largest. A snapshot is a moment, so restoring it
commits no frame past it. Old state under current laws and the old world
whole are different acts, so a restore names which.

**Where.** `Evaluator::restore` in the
[graph's state](../crates/core/graph/src/evaluate/state.rs);
`Session::restore` for the laws; `restore` in the protocol. Held by the
protocol's tests (a restored world counts on from the snapshot; one that
cannot run is refused with its laws taken back; one restored with its
laws runs as it ran) and by the session's random life.

## A platform that lacks a primitive does the same work another way

Under WASI (no threads, no resolved links, no code made at run time):
where no thread can start, its work is done in turn (an `async` run lands
before its start returns, Kernels are prepared inline, `serve` answers
each request before reading the next). Where links cannot be resolved, a
file's one name is its path made whole with `.` and `..` folded, and the
sandbox keeps that name honest. Where nothing loads JIT code (WASI under
a plain runtime) there is no JIT, as there is no GPU without a device,
and asking for it says why; where the host loads it (a page), the JIT
writes WebAssembly.

**Why.** A platform is the Host's world: its primitives vary, what a
scene computes does not. Each fallback is one a native run already
allows, so no document, replay or agent sees a different world.

**Where.** [`later.rs`](../crates/core/kernel/src/later.rs) and
`set_background` in the runtime,
[`store.rs`](../crates/interface/session/src/store.rs) in the session,
`stdin_into` in the protocol's transport. Held by the example replays,
run under `wasmtime` on the interpreter.

## A session's documents are wherever its caller keeps them

A session reads its scene, the documents it uses and the files its
inputs name, and writes its outputs and requests, through one `Store`:
the disk, or files its caller handed in and may hand in again. The C ABI
opens a session on handed-in files; a page running Orla as WebAssembly
does exactly that. A file handed in again is a changed file, read at the
next sync. Where a machine has no clock, nothing is timed and history is
not dated.

**Why.** Source resolution belongs to the caller or the file loader
(AGENTS.md). What `use` reaches, what a package holds, when a document
changed and where a write lands are one question in memory and on disk,
answered once; hot reload is the same act in a page.

**Where.** [`store.rs`](../crates/interface/session/src/store.rs) in
the session; `orla_files`, `orla_files_put` and `orla_open_files` in
the [C ABI](../crates/interface/ffi/include/orla.h);
[`clock.rs`](../crates/core/kernel/src/clock.rs) in the runtime;
`OrlaInPage` in the web bridge. Held by the ABI's test (a scene from
paths no disk has, a Kernel put again) and the web bridge's, which runs
the text example in a headless Chrome.

## Where Orla is WebAssembly, its JIT writes WebAssembly

Where Orla is WebAssembly, the JIT writes a Kernel's IR as a WebAssembly
module, instruction for instruction what the Cranelift translation does,
importing Orla's memory and function table. The host (a page's script;
node, for the tests) compiles it and puts its functions into Orla's
table; the index it returns is a function pointer, so the drivers of the
five modes, the context, the helpers, integer failures and the work
budget are the JIT's own, unchanged. What would be on a stack lives in
slots the prepared Kernel owns: WebAssembly runs on one thread and a
Kernel's functions do not recurse.

**Why.** Kernels arrive and change while Orla runs, so only compiling
them where they arrive makes them fast. A function pointer that is a
table index shares the native JIT's host side; the module is written by
hand, so no dependency is taken.

**Where.** [`wasm/`](../crates/backends/jit/src/wasm) in the JIT, its
`wasm-host` feature, which `orla-ffi` turns on for a page; `host` in
the web bridge's [`orla.js`](../bridges/web/orla.js), which
[`wasi.mjs`](../bridges/web/wasi.mjs) uses to run Orla's WebAssembly
tests in node. Held by the JIT's own tests (parity cases, feature
matrix, random programs) run as WebAssembly, alongside the core's, the
front end's and the interpreter's
(`crates/interface/orla/tests/wasm.rs`); and by the web bridge's test,
which runs the text and voxel examples in a headless Chrome.

## In a page, waiting for the device is giving the page a turn

Browser WebGPU answers a readback, a pipeline's validation, even the
device, only once the page's script has had a turn, and an evaluation
waits for such answers mid-frame (a JIT scan reading GPU output, a
`value(port)`, an async node's inputs, a `read`). In a page the GPU
backend waits by parking: Orla calls `orla.wait`, a suspending import
(JavaScript Promise Integration), and is suspended until the device's
answer, or a future's waker, calls `orla.wake`. Nothing spins; input
waits for the frame to end. The GPU backend is one everywhere; only how
it waits differs (`orla_gpu::Wait`: block, or park), and `orla-ffi`
supplies the turn. A parking wait submits an empty batch after it asks
for a mapping, since a browser may look at the device only when
something is submitted; a blocking wait polls the device. The browser is
reached through wgpu's WebGPU backend, whose glue `wasm-bindgen` writes;
its import of `orla` is the page's `orla-host.mjs`, through an import
map.

**Why.** Without suspension a frame would split at every place a backend
reads what another made, and the evaluator, protocol and Backend
contract would all learn to stop and go on. Suspending the machine is
the host's to offer; nothing above the GPU backend learns it was
suspended, and one GPU backend keeps one body of evidence.

**Where.** [`wait.rs`](../crates/backends/gpu/src/wait.rs) in the GPU
backend, `Session::runtime_waiting`, `waiting` in the
[C ABI](../crates/interface/ffi/src/lib.rs), `host` and `OrlaInPage` in
the web bridge's [`orla.js`](../bridges/web/orla.js). Held by the web
bridge's test, which runs the text example's checks and the voxel example
on the page's WebGPU in a headless Chrome.

## What playing a scene means is decided once

Every interface that plays a scene shares one play
(`orla_session::play`): the frame (changes read, a replay's frame taken
at its time, the scene evaluated, picture and sound named), what F5,
Tab, F10 and Escape mean, how a pad is a keyboard, how the pointer is
found in the picture, the line of frame data, and what is said. An
interface is its devices only: it hands the play its person's input and
clock, shows the picture and plays the sound on its own devices, and
tells the play where the picture landed.

What every interface asks of a session the session answers: which output
is the picture and which the sound (`Session::picture_of`, `sound_of`), a
Field or a picture read back (`Session::field`, `picture`), the backend
when none is named (`best`: the GPU, else the JIT, else the interpreter,
the ones passed over said), every command's session flags
(`SessionOptions`, `graph` too), and what checking a document finds
(`check_document`, through the store). Each interface drives its own
frames (a replay at its written times, a window at its clock, an agent
at the times it sends, `render` at a fixed rate); what a frame is,
`Session::evaluate`, is one.

**Why.** ARCHITECTURE-PLAN §34: interfaces are many, and "all of them
operate on the same underlying Runtime". A display, keys and a sound
card are the Host's IO (§6); a page is another Host, not another
product.

**Where.** [`play`](../crates/interface/session/src/play/mod.rs), with
its [pad](../crates/interface/session/src/play/pad.rs) and
[line of frame data](../crates/interface/session/src/play/hud.rs); the
window, sound card and pads of [the Player](../crates/interface/player/src/main.rs).
A page's Player is its devices over the same play, through the C ABI
(`orla_play_*`, [play.rs](../crates/interface/ffi/src/play.rs)):
[player.js](../bridges/web/player.js) hands it the page's keys, pointer,
wheel, pads and clock (`orla.now`). Held by
[the play's tests](../crates/interface/session/tests/play.rs), headless
(keys, preview, pointer, sound output, a failure said once, a replay
played to its end); by the ABI's test through `orla_play_*`; and by the
web bridge's test, a replay in `play.html` on the page's GPU and JIT, F5
and Tab pressed there.

## A replay is a test; a piece is made of laws, structure and state

A replay is input played into a scene over time, and what it expects: a
test. It lives under its example's `tests/`. Played headless (`orla
replay`), it holds the scene, frames as fast as they compute; played by
the shared play (`orla-player --replay`, `play.html?replay=`), it also
holds the interface (keys and pads mapped, pointer found, picture shown,
a page's device waited for) and can be watched. The window is handed
over when it ends.

A piece made with a scene is kept as what makes it: its laws as Kernels
(`publish`), its structure as the scene document (`compose` and its
`edits`), and its state as a port's Field (`read`, `write`, `save`) or
the whole world (`snapshot`, `restore`).

**Why.** A replay of clicks tests the editor clicked: change the editor
and the piece is gone. Headless and played are one replay and one play;
only played runs the interface, so a fault of the interface is found
there.

**Where.** [The replay module](../crates/interface/session/src/replay.rs)
reads and runs a replay; [`play`](../crates/interface/session/src/play/mod.rs)
plays one. Held by the example replays under the verification, the
Player's test that a replay played in a window holds as it does
headless, the ABI's through `orla_play_*`, and the web bridge's in
`play.html` on the page's GPU and JIT.

## A rule that had two owners is held to one

A rule found with a second owner (a syntax parsed in two places, a
second way to the disk, an interface asking the session's question) is
left with one owner and held there: by the compiler where visibility can
say it, by a source scan where it cannot. The session's store is its
own, so a caller reads or writes on another's behalf through
`Session::read_file`, `write_file` and `write_scene`, each checking what
may be before it acts; a caller that reaches the store does not compile.
A scan row names the act a copy would do (cut `node.port` at a dot,
write a file past the store), its owner, and the commit that gave it
one. A rule is scanned for only after it has had a second owner, and
only by an act that finds that owner in the source before the fix; a row
that fails on legitimate code is rewritten, never widened with
exceptions.

**Why.** A rule with two owners drifts until they disagree, and tests
hold behaviour, not ownership. A guard per broken rule is few and
precise, each proven by the bug it catches; a guard per rule is fragile
patterns readers learn to silence.

**Where.** [`owners.rs`](../crates/interface/orla/tests/owners.rs): the
rows, each shown to fail on the source its commit replaced. The
session's store, `pub(crate)` in [the session](../crates/interface/session/src/session.rs).
The layers are held beside it, in [`layers.rs`](../crates/interface/orla/tests/layers.rs).

## Orla describes itself, from the tables it enforces

Orla's syntax and API are one reference, layer by layer: the types, the
builtins and the forms of the Kernel language; the scene's forms and
what a `domain` may call; the capabilities and what each grants; a
package's parts; the protocol's operations and its wire; the replay
lines; the command line. Each part is rendered from the table the
implementation executes, so it cannot say what the implementation does
not do. `help` answers all of it, or one name in full;
`docs/REFERENCE.md` is the same text. Whole programs are the examples.

**Why.** A description the implementation cannot drift from is one
rendered from what it executes. An agent learns a language fastest from
one dense reference and whole programs; anything built beyond that to
teach it costs more than it saves. A doc lives in the source it
describes, so it travels with a Kernel through publish and snapshot; the
IR stays computation alone.

**Where.** [`help`](../crates/interface/protocol/src/help.rs),
[`ops`](../crates/interface/protocol/src/ops.rs),
[`envelope`](../crates/interface/protocol/src/envelope.rs),
[`Capability::doc`](../crates/core/abi/src/lib.rs),
[`replay::forms`](../crates/interface/session/src/replay.rs),
[`commands`](../crates/interface/orla/src/commands.rs),
[`doc`](../crates/frontend/language/src/doc.rs); held by
[the signatures compiled](../crates/frontend/compiler/tests/help_signatures.rs)
and [the forms parsed](../crates/frontend/compiler/tests/help_examples.rs).
