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

## A function never reaches itself; its place in the document is free

A Kernel's functions may call one another whatever order they are
written in, within a document and across the documents it uses. No
chain of calls may come back to a function it has left: `fact` calling
`fact`, or `a` calling `b` calling `a`, is refused where the chain
closes, naming the chain whole (`a -> b -> a`). Repetition is a `loop`
carrying the work.

**Why.** A Kernel has no stack. WGSL forbids recursion; the WebAssembly
JIT keeps what a stack would hold in slots its prepared Kernel owns; a
short function is inlined; and an element's work is bounded by its loops
alone. Refusing a cycle protects exactly that, where a rule of position
(a function calls only those above it) refused orders with no cycle and
made a document's order a thing to arrange: a caller written first, or a
helper added at the end by an edit, was refused for being where it was.
A function still reaches only its own document's declarations and those
of the documents it uses, never a document that uses it.

**Where.** `refuse_cycle` and `Shared::lowering` in the compiler's
[`lower`](../crates/frontend/compiler/src/lower/mod.rs): a function is
lowered where first called, with the functions being lowered on a stack,
and a call to one on it is the cycle. The IR keeps its own order,
callees before callers ([`Function`](../crates/core/ir/src/lib.rs)).
Tests `no_chain_of_calls_reaches_back_to_a_function` and
`functions_call_each_other_in_any_order_whatever_is_lowered_first`.

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

A run the Host has no room for, for a Field or for its pairs, runs once
more after the evaluation lets go of the cached outputs no later run of
the frame reads: those read last before it, and those downstream of a
node run this frame, which run again when reached. The scene's outputs,
the state's sources and the pinned stay. A frame of many steps holds
what its next steps read, not every step it took; a later frame that
asks for an output let go runs its node again.

**Why.** The Host owns memory as it owns work. An element that emits in
a loop holds as much as its work bound lets it, and that bound is on
iterations, not memory; a published Kernel must not take the machine's
memory by asking.

**Where.** [`Scope::room`](../crates/core/abi/src/lib.rs); `OverRoom` in
[`orla_rules`](../crates/core/rules/src/lib.rs), held by the JIT's
[parity tests](../crates/backends/jit/tests/parity.rs). Letting go is
`Evaluator::make_room` in [the evaluator](../crates/core/graph/src/evaluate/ledger.rs),
held by [a chain longer than the Host holds](../crates/interface/orla/tests/session.rs).

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
`cutout_window.osn` where a mask's alpha cuts fragments away;
`canvas.osn` over an empty canvas another node sized) and binds the
preset's fast path in its place, one node for another, where it wants
speed: `native.rasterize@1` for `raster.osn` and `window.osn`, `@2` for
the cutout presets, `@3` for `canvas.osn`. Both keep one rule: a pixel is covered where its centre is inside the
triangle's box and on or inside all three edges, holds the nearest
fragment in `[0, 1)`, the earlier triangle on a tie, with
attributes interpolated by `1 / w`; with a cutout, a fragment whose alpha
in the mask at its own uv is below the cutout is not drawn. The `tiled`
presets cut a box larger than a tile across a pool of invocations, for
scenes of large triangles; the surface is the same to the bit where
every Kernel rounds alike, as on the interpreter and the JIT. A
triangle with two corners at one point is skipped by that test alone.
Each triangle's box is found once, and every step reads it.

**Why.** As Kernels, rasterization is typed, traced, hot-reloaded, the
same on every backend, and changeable by an author. A native operation
is a package's Rust fast path, written where its Kernels are not fast
enough: each one is a preset of the package, its ports the preset's, so
every native has a Kernel definition it is held to. The native path is
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
the bit on the interpreter and the JIT; `canvas.replay` holds `canvas.osn`
to `raster.osn` to the bit and to `native.rasterize@3` on all three.

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
closed forms, [`many.replay`](../examples/physics/tests/many.replay)
the grid's pairs to every pair asked, and [`pile.replay`](../examples/physics/tests/pile.replay)
a pile some twenty deep settled in four substeps, none pushed out, on
all three backends. A pass that pushes every body reads bodies every
manifold has pushed: a warm is taken back in a pass of its own, not in
the bounce's, whose copies would each see their own share taken back and
none of the others'.

## A small run goes to the processor; a device is for many elements

When the main backend is a device and the JIT runs beside it, each run of
a Kernel goes to the one its size suits: to the JIT when its elements
times its Kernel's typical work (`Validated::typical_work`, an estimate
of operations per element from the IR) are at most `--small`, 400,000 by
default; to the device otherwise. A run of so few elements that one of
the device's groups holds them all (64, its 8 by 8) goes to the JIT
however much each takes, since the device runs them a thread each, far
slower than the processor. A scatter carries its target whole, so the
target's elements count among its own: one of few elements written into
a large target stays where the target is, rather than copying it to the
processor and back for its readers. A Kernel the device declines always runs on
the JIT. A small run that reads a Field the device made in this
evaluation stays on the device, since bringing it back would wait for
the device's work before it and break it in two; a run of so few
elements leaves it still when it is long, past 100,000 operations (two
thirds of a millisecond a thread at a time on the device, at 6.6 ns an
operation, more than the wait), unless what it would bring back passes a
mebibyte. A Field made before, state or an input, comes back once. A Field uploaded from memory keeps its words in memory
beside the device's copy, as a Field is a value, so neither side sends
it back again. The report says where each run went and why; `--small 0`
keeps every run on the device.

**Why.** A device's dispatch costs about the same whatever it does:
80 microseconds on the machine it was measured on, where a chain of two
hundred small runs takes 15.6 ms on the device and 0.73 ms on the JIT,
and a step of sixty-seven rigid bodies 10.6 ms against 0.57 ms. Submitting
them together, or in one pass, costs the same: the time is the dependent
dispatches themselves. A device earns its cost over many elements, and
nothing over few: a game's step of one element, a fight's whole frame,
took 0.96 ms on the device and 0.033 ms on the JIT, its readers of a few
elements kept on the device after it; the frame went from 11.7 to 10.0
ms. A short run between two of the device's, a learner's guess and its
error, costs more taken off it than left (3.32 against 3.93 ms a
frame), so
where a run goes follows its size, not whether it draws or computes. The
choice reads the run's elements and the Kernel, never a clock, so a
replay places every run the same way each time. A run may round its
last bits differently on the two, so a decision two Kernels share is
made in one (see *The rasterizer is a package*).

**Where.** `Runtime::execute` in [`orla-kernel`](../crates/core/kernel/src/lib.rs),
the estimate in [`orla-ir`](../crates/core/ir/src/lib.rs), the kept
words in [`orla-host`](../crates/core/host/src/lib.rs);
[`routing.rs`](../crates/interface/session/tests/routing.rs) holds a
scene of every kind of work to the JIT's alone with small runs placed
off a main backend, and the verify recipe replays every example on the
device with and without `--small 0`.

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
[`examples/data`](../examples/data/README.md).

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
F6 and F10 mean (every other key, Tab and Escape among them, is the
scene's), how a pad is a keyboard, how the pointer is
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
and F6 pressed there.

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

## A graph says what its parameters must hold

`require name: a <= b;` is a scene's statement of what its parameters
must hold to run: two counts, written as a `domain` writes them, from
the graph's params, numbers, `window()` and `scale()`, compared. It is
checked as the document loads, with the parameters as given, and before
every evaluation with the parameters as they stand; one that fails
refuses the document or stops the evaluation before any node runs,
naming the requirement, where it is written and what each side came to. A placed
graph's requirements come with it, reading what the parent passed. The
physics package requires its grid's cells to fit the bits its sort
orders; the sort requires 1 to 31 bits.

**Why.** A package's preset is a composition, and what breaks one is a
relation between its parameters (`columns * rows` against `bits`) that
no single Kernel sees whole and none can refuse: a Kernel has no way to
fail but a fault, and a fault in the middle of a frame names the wrong
thing. The relation belongs to the graph that states the parameters. It
reads counts and yields a verdict, so the graph still computes
structure, never values, and it is checked where the params are known,
once per evaluation, on the Host and on no backend.

**Where.** [`Requirement`](../crates/core/graph/src/domain.rs), checked
at the start of an evaluation in
[`evaluate`](../crates/core/graph/src/evaluate/mod.rs); lowered and
flattened in [`scene`](../crates/frontend/scene/src/lib.rs) (test
`requirements_flatten_named_with_their_place_reading_what_the_parent_passed`),
and held end to end by `a_requirement_the_parameters_break_stops_the_evaluation`
in [the session tests](../crates/interface/orla/tests/session.rs).

## The framework is tested by its own scenes, a demo by its replays

Rust tests hold the framework: the language, the IR, the runtime, the
backends and the session, each with scenes written in the test. None
reads, lists or names a file under `examples/` or `packages/`. A demo or
a package holds itself: its replays, run by the release `orla replay` on
every backend, with nothing left to the processor by its size, with
nothing fused, and rounded apart. A replay whose checks hold two forms
equal word for word says `alike`, and rounding apart leaves it out.

**Why.** A demo is written in `.orla` and `.osn`, and its author needs
no Rust to validate it. Rust tests that walk the demos tie the framework
to them: every demo added makes the framework's tests slower, and every
exception a demo needs lands in Rust. Replays in the release build hold
every demo in seconds, and a new one joins by being written.

**Where.** The recipe's `replays` step in [PROCESS](../PROCESS.md); the
`alike` line in [`replay`](../crates/interface/session/src/replay.rs).

## A name is one package; a package's names carry its name

A package is found by its name, and the name is a fact the manifest
declares and its directory carries: `voxel@1` is `voxel.orlapkg`. A
scene reaches one package of a name; two, as two editions of one, are
refused, naming both. A package names the packages it uses by name and
finds them beside itself. There are no qualified names: a document's
functions, structs and types are scoped by their document, so two
packages keep a helper of one name each, and only a name the scene
calls must answer once. A package's public names carry its name
(`voxel_walk`, `VoxelBox`), so a scene using many calls each plainly.

**Why.** A name that is checked is one a reader and a `use` agree on;
two editions of one package in one reach meet as near-alike
declarations, and refusing them by package says what an author can fix.
A qualified name would be a second form of every name, through the
language, the IR's function names, the protocol's reads and every
message; scoping by document already keeps packages apart, and a prefix
costs a few characters a public name.

**Where.** `package_manifest` and `one_package_of_its_name` in
[`sources.rs`](../crates/interface/session/src/sources.rs), with
`resolve_package` for the search beside a package; the scoping of names
in [`scope.rs`](../crates/frontend/language/src/scope.rs); the recipe's
check of every package in one scene in [PROCESS](../PROCESS.md).

## A document sees what it uses itself

A document's names are its own declarations and those of the documents
its `use` items name: each file of a directory, each file a package's
manifest uses. What those documents use in turn is theirs: their
functions call it, and a document that wants it names it in a `use` of
its own. A name it reaches only through another's `use` is refused,
said with the `use` that would bring it.

**Why.** Seen through another document, a name is a dependency no line
of the document states, and a coupling it did not choose: a game's
widgets using a package brought every name of the package into each
document that used the widgets, and ten of the game's own names clashed
with the package's though none of those documents used it. Names are
then what Kernels already were: a scene publishes the Kernels of the
documents it uses itself. No form is added: a `use` is how a document
says what it depends on, and now the only way.

**Where.** `closure` and the views in
[`scope.rs`](../crates/frontend/language/src/scope.rs); the hint in
`located` in [`sources.rs`](../crates/interface/session/src/sources.rs);
`a_document_sees_what_it_uses_itself` and
`names_a_used_library_uses_are_not_the_documents` in
[`sources` tests](../crates/interface/session/tests/sources.rs).

## Sources belong to the loader; names belong to the language

The loader finds documents: it resolves a `use` to files and packages,
refuses cycles and two packages of one name, and hands the language the
documents it reached and which uses which. The language decides every
name: a Kernel sees its own declarations, then its document's, then what
the documents its `use` items name declare, one declaration a name (alike ones
are one), and the builtins, which a function of a builtin's name hides
in its own document only. A declaration a Kernel inherits carries where
it was written and the name it was written with, so a fault is placed
and said by looking them up, never by taking a name apart.

**Why.** The rules of names were split between the session, which
renamed across documents, and the language, which renamed within a
Kernel, each writing what the other needed into the names themselves.
The seams between them were where alike declarations were merged twice,
where a fault was placed in the wrong file, and where a library's
function changed what a builtin meant in the documents that used it. One
owner of names, and origins as data, leave nothing to take apart.

**Where.** [`scope.rs`](../crates/frontend/language/src/scope.rs):
`document_scope`, `scope` and `resolve`; `Origin` and `Held` in
[the language](../crates/frontend/language/src/lib.rs); the walk of the
`use` items, `declarations_within`, in
[`sources.rs`](../crates/interface/session/src/sources.rs).

## A replay played again computes only what changed

A replay is played again as its documents change, and most of each of
its frames reads nothing that changed. A run played under a name is
remembered frame by frame: each node's cache entry, its key (the Kernel's
generation, parameters, inputs by their producers' epochs, domain, clock
and input) and its outputs. Played again, a node whose key is the one it
had on that frame takes those outputs; a node that runs and computes
outputs equal word for word to the remembered ones takes their epoch, so
what reads it finds its key unchanged. Outputs too large to hold are
remembered by key and epoch alone, and a node whose key matches runs
again under that epoch when its backend repeats exactly; on any other
backend it runs under a new one and its readers run. Runs are held to a
budget, forgotten when the graph changes, and give way when the Host is
short of room.

**Why.** The key names everything a node's outputs depend on, so held
outputs under an equal key are the outputs; taking them is not an
approximation and a replay played so passes or fails as a fresh one
does. Outputs not held are trusted to be made again word for word only
where the backend says so, and its conformance tests hold it to that, so
the rule rests on a stated contract, not on how a backend happens to be
written. Remembering whole frames by trajectory, not by a hash of
the world, keeps the rule the cache already keeps between one frame and
the next.

**Where.** [`recall.rs`](../crates/core/graph/src/evaluate/recall.rs)
remembers and offers runs; [`Resident::play_as`](../crates/interface/session/src/replay.rs)
names a replay's run; held by [the recall tests](../crates/interface/session/tests/recall.rs),
which compare a run played again with a fresh session, and by each
backend's run of the conformance matrix twice
([`unrepeated`](../crates/backends/conformance/src/lib.rs)).

## A Kernel a device cannot build runs where it can

A device's own compiler can fail on a shader that is valid: a large one,
for no fault in it that the backend can name (a crash in its register
allocator, as one was found). That is not a fault of the document, so it
does not refuse the document: the backend learns, in building it, that
it cannot run that Kernel, declines it from then on as it declines what
it cannot run beforehand, and the runtime prepares it on the ordered
backend, said once as any Kernel off the main backend is. A shader the
backend emitted that the device refuses as invalid is the backend's own
fault and still refuses the document, so no fault of Orla's is hidden by
running elsewhere. The device's errors are caught whatever their kind,
never left to panic.

**Why.** Placement is a backend's to decide and the document's to read
about, never to fail on; the JIT computes what the reference does, so a
Kernel moved there answers the same. A large game is made of large
Kernels, and one compiler's limits must not stop it.

**Where.** [`Scopes`](../crates/backends/gpu/src/failure.rs) catch the
device's errors, sorted into the backend's own and the device's; the GPU's
[`declines`](../crates/backends/gpu/src/lib.rs) answers for what its
device could not build; the runtime's `prepare_on` routes it
([kernel](../crates/core/kernel/src/lib.rs)); held by
[the unbuilt test](../crates/interface/session/tests/unbuilt.rs).
`orla check` says when a Kernel is large and what makes it so
([size](../crates/core/ir/src/size.rs)); see *A Kernel is large by what
it carries*.

## A Kernel is large by what it carries

`orla check` judges a Kernel's cost to a device's compiler by the words
it carries through its branches: every struct value an `if`, a `match`
or a `loop` yields, its words, each function counted as often as it is
called, as such a compiler writes every call out. Past 5 000 it says the
Kernel is large, naming the structs and the functions that carry most.
Its instructions are not the measure. A scene that publishes Kernels no
node uses is told which, as each is compiled and prepared all the same.
A running session says both where its author works: a large Kernel with
each version it publishes (so a change that narrows it shows the words
falling), the unused ones as the scene loads.

**Why.** Measured on Metal over every Kernel of the examples and a large
game: the Kernels its compiler took seconds over carried 7 000 to
31 000 words, and every other carried under 4 200; the instruction count
put a dozen fast Kernels among them. A compiler keeps a merged value's
words live together, and takes longer past in proportion to how wide
they are: a 220-word struct through forty branches took twelve times as
long as a 40-word one, and a struct kept apart, changed apart, half as
long as the same words nested in one. What makes a Kernel slow to build
is what a game is most tempted to write, one row of all its state
updated everywhere; the note names that, so a narrower struct, or state
in several rows, is the author's to choose.

**Where.** [`carried`](../crates/core/ir/src/size.rs); `large` and the
unused note in [check](../crates/interface/session/src/check.rs), and the
session's `Large` and `Idle` events
([session](../crates/interface/session/src/session.rs)); held by
[the large test](../crates/interface/session/tests/large.rs).

## A device builds a Kernel when it is needed

Preparing a Kernel for a device validates it as the device takes it (its
shaders, its bindings) and builds nothing: the device's own compiler, the
long part and the part that fails, runs when the Kernel is needed there.
A session builds the Kernels its graph's nodes use for the first time as
it publishes them, on every core at once, so the first frame waits for
none of them in turn; a replacement for a version that ran on the device
is built before it is ready, and one for a version whose runs the size
sent to the ordered backend is prepared there before it is ready too,
so the running version keeps the frames
until the new one can take them without a wait, replacements prepared on
every core but one, which the frames keep; any other Kernel is built
by the first run that needs it. A build that fails is a Kernel the device
cannot build: it goes to the ordered backend from then on, a run that
found it made again there.

**Why.** A document publishes more Kernels than a scene uses, and a scene
runs many of its Kernels where the device never does (a few elements, a
small run); building them all made every edit wait for compiles nothing
read, one compile after another. Validating first keeps the rule that
nothing invalid publishes; building where it is needed keeps hot reload
from stalling a frame.

**Where.** The GPU's [`Lazy`](../crates/backends/gpu/src/prepare.rs) and
`Backend::finish` ([backend](../crates/core/rules/src/backend.rs)); the
runtime's `prepare_finished`, `Preparer::finish` and a version's `leaves`
([kernel](../crates/core/kernel/src/lib.rs)); the session's `finish_used`
([session](../crates/interface/session/src/session.rs)); held by
[the lazy test](../crates/interface/session/tests/lazy.rs).

## A changed document compiles off the frames

Where replacements prepare off the frames (a Player, a served session),
so are library documents read before and changed since read and their
Kernels compiled (one read for the first time is read at once, as a
Kernel nothing runs yet prepares at once): a sync
hands them to another thread, reading them on every core, and returns;
the running Kernels keep the frames; the first sync after it is done
publishes them in order, or says why not, as a sync that read them itself
would. The scene document is read once they land, so it never names a
Kernel they are about to publish. Without a worker (a replay, a check)
a sync reads them on every core and publishes them itself.

**Why.** Reading a document and compiling its Kernels is the frontend's
work, and a shared document reaches many: done on the frame it stopped
the frames for most of a second. Moving it changes when a change
publishes, never what publishes: nothing invalid publishes, and a change
publishes whole or not at all.

**Where.** `read_library`, `Compiling` and `Scene::sync_libraries`
([scene](../crates/interface/session/src/scene.rs)); `Ledger::prime`
([kernels](../crates/interface/session/src/kernels.rs));
`Session::sync_settled` waits for it; held by
[the compile-ahead test](../crates/interface/session/tests/compile_ahead.rs).

## The language grows only by the owner's word

A form, a keyword, a builtin or a shorthand is added to `.orla` or `.osn`
only when the owner asks for it or agrees to it in words. A contributor,
an agent above all, who finds writing hard meets it with what the
language already has: a function, a package, or an example that shows
the way. A proposal to grow the language is put to the owner, answering:

1. It lowers to the IR there is: no instruction, backend behaviour,
   budget rule or runtime concept is new.
2. How often sources in and beside the repository write what it stands
   for, counted, not felt while writing one demo.
3. Why a function, a package or the forms there are cannot write it, or
   write it only wrongly.
4. It is a word or a builtin everywhere it may appear, never one that
   means something in one place alone.

**Why.** Every form is one more thing each creator, an AI above all, must
learn, and one more thing every later rule must hold for. Friction met
by one demo at a time grows a language by the sum of its writers' local
pains, which no one decided. Orla builds the conditions features come
from, not the features; a function or a package is such a condition, a
shorthand for one pattern is not.

**Where.** [`surface.txt`](../crates/frontend/compiler/tests/surface.txt)
holds every builtin and form; [`surface.rs`](../crates/frontend/compiler/tests/surface.rs)
fails when it grows and names this entry; the scene syntax is held by
[the scene documents' surface](../crates/frontend/scene/tests/surface/).
`AGENTS.md` and `PROCESS.md` state the rule where a contributor reads
first.

## A memory is carried page by page

A Field keeps the extent it was declared with; that extent is its
ceiling, and what a run costs is the pages it changes. The Host divides
a Field's words into pages of `PAGE_WORDS`. A Field made from another
records the other's words, as a `Version` (the Field and how often it
was written in place), and which of its pages differ
(`Host::adopt_derived`), and so for what those were made from, up to
`ANCESTRY` back, the pages since added up: a frame's chain of scans
reaches the frame before. A Field written in place has a new version, so
nothing made from its earlier words takes them for its present ones. A
released Field's buffer is kept on the Host's spare pile marked with the
version it holds and what it was made from, so `Host::working` hands
back a Field's words from the buffer of words it was made from, of a
Field made from it, or of one made from the same words, copying the
pages either changed alone: a scan's memory bound to the same empty
Field run after run is made from the buffer its last run left. `Host::same`
compares two Fields on those pages the same way. The JIT carries a scan's memory this
way: it marks a page when a write changes a word on it, and a run it
cannot follow word by word is adopted whole, as any Field is. The GPU
brings such a Field up the same way: from the buffer an upload of
words it was made from left behind, or of other words made from the
same, writing the pages either changed, else from a device copy of a
live base's buffer. The interpreter copies a memory whole, the
reference the others are held to; a backend that does not record what it
changed gives up nothing but the saving.

**Why.** A scan's memory is sized once, for the most it may hold, and
the most is what a creator must declare. Copying, comparing and
uploading it whole on every run made each edit cost the ceiling, not the
edit. Allocating by need would change what a Field is and what a budget
counts; pages keep the Field a value of the extent it says, the budget
counting it whole, and make only the work proportional to the change.

**Where.** `Pages`, `Version`, `Host::working`, `Host::adopt_derived`,
`Host::made_from` and `Host::same` ([host](../crates/core/host/src/lib.rs)),
through the `Scope` ([abi](../crates/core/abi/src/lib.rs)); the JIT's
`Changed` ([jit](../crates/backends/jit/src/lib.rs)); the GPU's `Held`
and `Spare::take_made` ([pool](../crates/backends/gpu/src/pool.rs)) and
`bind_input` ([stage](../crates/backends/gpu/src/stage.rs)); the
evaluation's `same_words` ([evaluate](../crates/core/graph/src/evaluate/mod.rs)).
Held by the Host's page tests,
[the carried memory parity test](../crates/backends/jit/tests/parity.rs)
and [the GPU's page upload test](../crates/backends/gpu/tests/parity.rs).

## A window's frames overlap the device's

The Player's evaluations overlap: a frame commits its state without
waiting for the device, and is confirmed at the end of the next frame,
once that frame's work has gone to the device behind it: its checks are
waited on, and a mark of its submitted work (`Backend::submitted`). A
frame found failed takes back its commit and the next frame's, which was
made from it; the state is what the frame before them left, the failure
is the answer of the call that found it, and the world's count of frames
drops the one answered as committed. Anything other than the next frame
confirms first: every other entry point of the evaluator, a request to
the Player (whose `evaluate` answers a confirmed frame), a file write of
a changed output, a save. What the processor takes home from a frame in
flight is queued behind its work and taken once the frame is confirmed
(`Runtime::fetch`): the Player hears each frame's sound a frame later. A
small node compared on the device, found unchanged while every node that
reads it ran all the same, rests as one found changed does: such a
comparison waits for the device and spares nothing. `orla serve`,
replays and the FFI do not overlap; the protocol's contracts are those
of a frame confirmed.

**Why.** A frame that waited for its own device work left the processor
idle while the device drew, and the device idle while the processor made
the next frame. Committing ahead keeps one frame in each, and holding
what the commit replaced keeps a failure as harmless as before: nothing
is released until the frame is confirmed. Two frames in flight at most
bound the device memory held as one frame in flight did. Any wait in
the middle of a frame (a download, a comparison) waits for the frame
before it as well, so what is read is read after confirmation, and a
comparison is kept only where it spares work.

**Where.** `Evaluator::set_overlap`, `confirm` and `commit_in_flight`
([frames](../crates/core/graph/src/evaluate/frames.rs)), `weigh_alike`
([evaluate](../crates/core/graph/src/evaluate/mod.rs)),
`Runtime::fetch` ([kernel](../crates/core/kernel/src/lib.rs)),
`Backend::submitted` and `Backend::download_later`
([rules](../crates/core/rules/src/backend.rs)), the GPU's waits bounded to
a submission ([wait](../crates/backends/gpu/src/wait.rs),
[later](../crates/backends/gpu/src/later.rs)), and the Player. Held by
[the overlap tests](../crates/interface/session/tests/overlap.rs) and
[the GPU's comparison tests](../crates/interface/session/tests/gpu.rs).

## A frame's device time is the device's to tell

The Player's frame-data line shows the frame's interval, the processor's
work on it (`cpu`, its waits for the device left out), and how long the
device was busy with it (`gpu`), and under it the nodes that kept the
device busiest. The device's time is taken without waiting for it: while
the line shows, every run the device is given, work or copy, is noted as
it is submitted and the device tells when it has done the run
(`Backend::set_clocked`, `clock_mark`, `end_frame`, `clocked_frame`); a
run kept the device busy from its submission, or the end of the run
before if later, to its own end. A node owns the runs given between the
marks taken before and after it runs (`Report::clocked`); a frame, told
some frames later by its number, gives each node its runs' sum, and the
runs no node owns are counted between nodes. A waiter on its own thread
hears the device as it finishes, so an end is told when it happens.
Nothing about how frames run changes while they are clocked, and nothing
is clocked while the line is hidden; over the protocol `evaluate` clocks
with `clock` and answers `device`.

**Why.** A reading that waits for the device changes what it reads: the
line once settled the device every frame to time it, which stopped the
frames overlapping, and a wait taken as the device's time measures only
what overlapping did not hide. A device's own timestamps would be
better, but a driver may take them and resolve nothing; every device
says when work is done. With `frame`, `cpu` and `gpu` side by side, the
frame follows whichever of the two is the longer.

**Where.** `Frames` ([timing](../crates/backends/gpu/src/timing.rs)),
`Backend::set_clocked` ([rules](../crates/core/rules/src/backend.rs)),
`NodeClock` ([clocked](../crates/interface/session/src/clocked.rs)),
`Stats` ([hud](../crates/interface/session/src/play/hud.rs)) and
`Play::frame` ([play](../crates/interface/session/src/play/mod.rs)). Held
by the clocked frame tests ([gpu](../crates/interface/session/tests/gpu.rs)).

## A frame runs what something reads

A frame runs the nodes something reads: every output's (one written to a
file too), every state's (what a `<~` carries and the source it starts
from), every node pinned to be read (a replay's `expect` and `show`, the
Player's preview, the protocol's `pin`, a `--save`), and what each of
them reads through `<-`, in turn (`Graph::read_through`). A node nothing
reads does not run: it keeps no value, `read` says so and how to give it
one, and `orla check` notes it once, at the top of its chain. A trial
(`compose` proving a scene before it lands) runs every node, so a node
not yet read is proved all the same.

**Why.** A Kernel is pure: a node nothing reads changes nothing anyone
sees, so running it is only its cost, every frame; a scene's dead
branches made it slower without a sign. What a frame costs is now what
it shows. Proving a scene is another matter: there, a node is run because
it may be read once it lands.

**Where.** `Graph::read_through`
([graph](../crates/core/graph/src/lib.rs)), `Demand::Frame` in the
[evaluator](../crates/core/graph/src/evaluate/mod.rs), the check's note
([check](../crates/interface/session/src/check.rs)) and `why_no_value`
([observe](../crates/interface/session/src/observe.rs)). Held by
[the unread tests](../crates/interface/session/tests/unread.rs).

## A check refuses what the types refuse

`orla check` holds every node, a native operation's too, to the rules a
binding and a run are held to wherever the types alone decide them: each
input of its port's shape and element layout, and the domain of the shape
its output (a scatter's first input) is. A source's type is its
producer's output port, or the shape an input read from a file is
declared, with its layout too where the element is one of the language's
own; a domain's shape is found as resolving it finds it, from the shapes
of the ports it reads and the parameters it names, or is its first
input's. What the data decides (a `value(port)` past the end, a count
that comes out zero) is the evaluation's. A check says a fault in the
words the evaluation would.

**Why.** A fault the types decide is a fault before anything runs; found
only by running, it is found late, on one backend, in another's words. A
rule said twice drifts: the binding's input check and the run's were two,
and said the same fault two ways, so they are one function each now,
which both call.

**Where.** `type_fault` ([bind](../crates/core/graph/src/evaluate/bind.rs)),
`DomainSpec::shape` ([domain](../crates/core/graph/src/domain.rs)),
`input_type_fault` and `check_domain_shape`
([rules](../crates/core/rules/src/lib.rs)), the check's sources
([check](../crates/interface/session/src/check.rs)). Held by
[the binding tests](../crates/interface/session/tests/binds.rs).

