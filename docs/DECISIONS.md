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
every element runs; it fails either way. Both bounds are a node's own: a
Kernel whose work is counted is never fused with another, so whether a
scene fits is not turned by fusion, nor by pinning a node to look at it.

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

## What a run holds in Host memory counts against the Host's budget

A scatter's pairs — the one each element returns and every one it
`emit`s — are held in Host memory by the interpreter and the JIT while
the run applies them, and they are held against the room the Host's
budget has left when the run begins
([`Scope::room`](../crates/core/abi/src/lib.rs)), as a Field is. The
pairs returned are held before the run starts, the ones emitted as they
come, and the JIT's copies sorted into its parts before it sorts them.
Past the room, the run fails as `OverRoom`, named for the room it had;
nothing commits.

**Why.** The Host owns memory, as it owns the run's work. Before
`emit`, what a scatter held was as large as its sources, and the Fields
it read already bounded that. An element that emits in a loop holds as
much as its work lets it, and the work bound is iterations, four
thousand million by default: a bound on time, not on memory. A Kernel an
agent publishes must not be able to take the machine's memory by asking
for it.

**What it does not cover.** A device that lands each pair where it is
emitted (the GPU combines it into the target with an atomic) holds no
list of pairs, and its pair of each element lives in device memory: a
run that fits there may not fit on the interpreter, which has less room
for the same work. Which backend a run fits on is a property of the
machine, as it is for any allocation; what a run computes is not. A run
that passes the room and also has a failing element may be named for
either, as a run past its work may.

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
| `.field` | [`field_file.rs`](../crates/interface/formats/tests/field_file.rs): every scalar and a spread of encodings pinned byte for byte, and every file in the repository still read |
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
surface — `raster.osn` or `window.osn`, and `cutout.osn` or
`cutout_window.osn` where a mask's alpha cuts fragments away — and binds
`native.rasterize@1` or `@2` in its place, one node for another, where
it wants speed. Both keep one rule: a pixel is covered where its centre
is inside the triangle's box and on or inside all three edges, holds the
nearest fragment in `[0, 1)`, the earlier triangle on a tie, with
attributes interpolated by `1 / w`; with a cutout, a fragment whose
alpha in the mask at its own uv is below the cutout is not drawn.

**Why.** Written as Kernels, rasterization is what the rest of Orla is:
typed, traced, hot-reloaded, and the same on every backend, edges
included, because every backend runs the same IR; and a rasterizer in
Rust is a feature an author cannot change. The fast path stays, as an
acceleration of the package rather than a second definition of it,
because it is faster where it matters and costs little to keep:

- On the interpreter, the reference every test leans on, the package
  draws a frame about three times as slow (the torus, from nine thousand
  triangles to two million: 2.9 to 3.3 times; the culled world 2.4; the
  model 3.1).
- On the JIT, 1.1 to 2.4 times as slow, the most where the rasterizer is
  most of a frame (a small mesh).
- On a GPU the fast path is the device's rasterizer, which the documents
  call approximate: the package is within 1.1 times of it on meshes of
  small triangles, and 9.6 times as slow on the model, whose few large
  triangles each take one invocation the whole of.
- Taking it away on the CPU and the JIT would make `native.rasterize` a
  device's alone, and a scene that named it would stop running there;
  choosing the package for it would need the Graph to know that a placed
  graph and a native operation are one computation, which it does not.

Measured on an M4 at 320 by 240 (the culled world at 640 by 400, the
model at 640 by 480), 2026-09-24.

A large triangle is the package's own weakness on a device: one
invocation walks its whole box. The `tiled` presets cut a box larger than
a tile across a pool of invocations, which draws the model on the GPU in
1.7 ms instead of 9.5 (the device's rasterizer: 1.0). They are not the
default because the cut costs every triangle a pass, a prefix and a pass
more, which a mesh of small triangles pays for and gains nothing from: the
torus of nine thousand triangles took twice as long, the culled world 1.4
times, two million triangles on the JIT 1.9 times. A scene with large
triangles names the tiled preset; the surface is the same to the bit.

At scale the package is the faster of the two on a device. Ten million
triangles at 4K ([`ten_million.osn`](../examples/mesh/tests/ten_million.osn),
the torus at `level = 6`, 9.4 million, at 3840 by 2160) draw on the GPU
in 19.7 to 20.0 ms a frame with the package and 32 with the device's
rasterizer, which writes a seven-word surface a pixel where the package
keeps a key of two. What had cost the package was the engine around its
Kernels, not the Kernels: an emitting scatter kept a list of the pairs
it returned and applied them in a pass of their own, and a target
compared as ordered keys was copied from its base and then encoded in a
second pass over it. Both are now one pass (the returned pair lands with
the emitted ones; the base is encoded as it is written into the target),
and the device's pile of spare buffers holds a gigabyte, the half of it
a frame at 4K asks for, rather than making some anew every frame. A
stream of screen positions worked out once a vertex, for the cover to
read instead of whole vertices, was measured and lost: its own pass cost
more than it saved (27 ms against 21). The frame is near what its memory
traffic allows — about two gigabytes, `project` and the cover's reads
most of it — and varies by half a millisecond from run to run, M4,
2026-09-24.

**Where.** The package and its presets; every example places it. On the
interpreter and the JIT the fast path is `orla_rules::raster`, held bit
for bit to the rule by a test that writes the rule plainly; on the GPU it
is the device's rasterizer. `examples/mesh/tests/edges.replay` pins the
package's surface at edge pixels on all three backends and `native.replay`
the fast path's inside them; `cutout.replay` holds the package's cutout
to `native.rasterize@2` on all three, and a session test holds them equal
to the bit on the interpreter and the JIT.

**What would change it.** A GPU fast path exact to the rule (a compute
rasterizer as fast as the device's, or integer coverage the device
honours), which would make the fast path a pure acceleration on every
backend; the package drawing a large triangle across many invocations,
which would close most of the gap on the GPU; or a scene that needs the
fast path chosen for it, which would need the Graph to know that a placed
graph and a native operation are one computation.

## A Kernel's effects are data the Host carries out after a commit

A Kernel never writes a file. A scene declares that one of its outputs is
also written to a file, `output kept <- state.result to "saves/state.field";`,
and the session writes it once an evaluation that changed it has
committed: whole (written beside and renamed over), into a directory the
person running the session allowed with `--allow file.write=<dir>`, and
nowhere else. Reading needs nothing new: an input already reads a file
and reads it again when it changes, and one that may be absent falls
back on a source meanwhile (`input saved: … = "saves/state.field" else
fresh.result;`). When and what to save is the Kernels': an output that
changes only when a Kernel says so is written only then.

**Why.** An effect inside a Kernel would break what makes a Kernel one:
the same words on the interpreter, the JIT and a device (a device has no
files); an evaluation that fails committing nothing (a file written by a
node before a later node failed cannot be taken back); and a replay being
the same every time. As data, the effect is a Field like any other,
typed and traced, the three backends compute it alike, and the Host
carries it out at the one moment nothing can undo: after the commit. The
plan's `requires file.write` is the allowance, given by the person
running the scene rather than claimed by the Kernel, so a generated
Kernel or a connected agent gets no reach it was not handed; a server
takes its allowances from the command line that started it. The session
decides every write, not the interface that asks: `save` and `export`
check its allowance whoever calls them and write through its store, and
a file the person names on a command line (`--save`, `--out`) is
allowed alone, not the directory it is in.

**Where.** [`writes.rs`](../crates/interface/session/src/writes.rs) in the
session; `to` and `else` in the document
([parser](../crates/frontend/language/src/parser.rs),
[lowering](../crates/frontend/scene/src/lib.rs)); a fallback in use
substitutes for its input when the evaluation starts
([evaluator](../crates/core/graph/src/evaluate/mod.rs)). Held by
[the session's write tests](../crates/interface/session/tests/writes.rs),
[the lowering's](../crates/frontend/scene/src/lib.rs) and
[`examples/save`](../examples/save/README.md).

**What would change it.** An effect whose target the Kernel must choose
while running (a file named by a Field, a network request), which would
make the target part of the data as well; a second effect beside files,
which would get its own allowance; or an effect that must happen before a
frame commits, which this model refuses on purpose.

## What a served scene reaches is what the person running it gave

A session an agent drives, `orla serve`, the Player's `--serve`, or a
program's `orla_request`, holds the scene to what the person who started
it allowed, and nothing that arrives over the wire widens it. Whether
reads are held is decided when the session is opened, before its scene
is first read, and never after: what may be read is the directories of
the scene as it first loads, whoever edits it later and whenever. A
program embedding Orla allows more with `orla_allow`, as a command line
does with `--allow`. An agent's `save` and a scene's `to`
land only where `--allow file.write` says. An input reads a file only
beside the documents the scene reads (the root, the graphs it places,
what it uses), inside a package, or where `--allow file.read` says. A
browser's page opens the socket only from this machine, from a file, or
from an origin `--allow-origin` names, and an HTTP request that is no
WebSocket upgrade is answered and closed. A Kernel's capabilities are
granted, not taken: what it `requires` must be among what the runtime
grants (every capability, unless `--grant` names fewer) for it to be
prepared at all, and it runs with what it requires of those. A Kernel or a scene an agent
sends is bounded before it costs anything: a domain is charged against
the Host's budget before it is allocated, on the frame or off it, an
`[each]` and a `times` before they are written out, a request on a
socket before it is held, and a rejected scene never reaches the disk.

**Why.** An agent composes the scene and reads back what it holds, so a
file it could read is a file it could send away, and any web page the
person visits can reach a socket on their machine. A scene a person runs
themselves (`run`, `render`, `replay`) reads what they chose to run, as
any program they start does, and has no channel back out but the
directories they allowed writes into; holding its reads too would only
make every test scene name its own assets again.

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

**What would change it.** A second person on the same session, which
would want who asked, not only what; a network listener beyond this
machine, which would want the socket itself authenticated; a
capability with an effect outside the Host, which would want its grant
to differ by who published the Kernel (a document the person wrote, or
an agent over the wire) rather than one set for both.

## Float `min` and `max` have one rule, and it is not the machine's

`max(a, b)` of floats is IEEE 754-2019 maximumNumber: a NaN gives way to
the other operand, and of two equal values the one without a sign, so
`max(-0.0, 0.0)` is `0.0` in either order; `min` is the same rule the
other way, `min(-0.0, 0.0)` is `-0.0`; `clamp` is `max` then `min`, and
`smoothstep` clamps so. The interpreter writes the rule out
(`orla_rules::max_f32` and its siblings), the JIT emits it, the GPU
calls it (`max_ref` in the shader's preamble).

**Why.** The reference had no rule: Rust's `f32::max` leaves the zero's
sign to the machine, and on this one the interpreter and the JIT gave
`inf` and `-inf` for the same program. Random programs found it
(`crates/backends/jit/tests/fuzz.rs`); the device's own `min` lost the
sign too. A semantics a backend may choose is not one.

**Cost.** A few instructions a call on the JIT: a scene whose hot loop
is mostly `max` (`examples/million`) runs about five per cent slower
there; the GPU's scenes measured within noise.

**What would change it.** An operand known not to be NaN or zero, a
constant, lets the check go; that is an optimization of the rule, not a
change to it.

## An integer sum is its whole total

A scatter's or a prefix's `sum` of integers is each element's whole
total: what it held and every value added to it, as mathematics adds
them. It fails only when that total is not one the scalar holds, and
names the first element whose total is not. A total that passes the end
of the range on the way and comes back is the total it comes back to.
The interpreter and the JIT keep each total wide (`orla_rules::Sums`);
the GPU adds a word atomically and what it carries into a high word
beside it, checks once every value is in, and, where a float is summed
beside the integer in element order, carries the high word as it folds.

**Why.** The reference added in element order and failed at the first
partial sum past the end; the GPU adds in whatever order its atomics
land, and wrapped. One IR had two meanings, and a counter or a
histogram passed on the device with a wrong value. A sum's inputs,
not the order a machine adds them in, decide it now, so every backend
answers alike, to the element that fails.

**Cost.** On the GPU, a high word beside each word a sum totals, one
more atomic where a value carries, and a pass that checks the totals;
on the CPU and the JIT, a wide total for each integer the target holds,
counted against the run's room.

**What would change it.** Nothing short of changing what `sum` means;
a `Long` summed on a device that has 64-bit atomics could keep its
totals the same way, a word wider.

## Reality changes atomically; history records why

A session's position is `(revision, frame)`: the revision moves when the
world changes other than by running (a Kernel published or retired, the
scene read again, a param set, state written or reset), the frame counts
the evaluations committed within it. Every change is recorded with where
it came from (its document, the protocol and its channel, or the program
itself) and the Kernel's sources are kept once each by digest, so an old
generation can be read and published again; the attempts refused are
recorded apart, with why, and move nothing. A snapshot takes the world
whole, committed state only, with the sources of the laws it ran under,
and is kept by whoever asked for it, or written to a file where writes
are allowed.

**Why.** An agent changing a live world needs to know how it came to be
as it is, and what was tried and refused; before, a replaced Kernel was
gone. The frame is its own coordinate because most scenes change their
state every frame: a revision per frame would bury the changes that are
discontinuous under the ones that are not. Only committed state is
taken because the cache may hold what a failed evaluation computed and
never committed; a snapshot of that is a world that never existed. The
session remembers changes, bounded, and never worlds: the past is data
its holder keeps, so the session stays the present.

**Where.** [`history.rs`](../crates/interface/session/src/history.rs) in
the session; `history`, `kernel` with `generation` and `snapshot` in
the protocol. Held by the protocol's tests and the history's own.

**What would change it.** A second person on a session, which would add
who to where.

## A live world is changed through three doors, each at its own grain

The structure changes as its document does, a statement at a time:
`compose` edits the scene text where a node, a member of a node, an
output or a declaration is written, and only there, or writes it whole;
the scene is then read again, as an edit on disk is, and a text whose
graph cannot run a frame (a source not there, a cycle) is refused before
it replaces the one running. The state
changes a committed port at a time: `write` replaces one, `reset` with a
`node` and `port` starts one again from its initial source, `reset`
alone starts them all, and `restore` installs a world. The parameters a
running graph takes from outside are the ones its document declares
(`param` in the scene); a node's own value is part of the structure and
changes with the document. Nothing else reaches in.

**Why.** Each door is where the thing it changes is kept whole. A node's
value set from outside would be a structure the document does not say,
lost at the next read and invisible to `document` and history. The
document is also a person's: its comments and layout say what its
structure is for, so an agent names what it changes and the rest of the
text is left as it was, rather than written again from the agent's
reading of it. State is
kept by port, so a port is the smallest part of it that can start again
without inventing a state no evaluation committed; a port restarted
alone is what an agent wants when one part of a world went wrong (a
camera, a score) and the rest (a built world) is the work. A run off the
frame that began from the state forgotten, or from what was computed
from it, is let go: landing after the reset, it would put back what was
forgotten. Runs that read none of it go on.

**Where.** [`edit`](../crates/frontend/scene/src/edit.rs) in the scene,
over the language's `outline` (where each statement and member was
written), and `compose` in the protocol; the graph's own check
(`Graph::order`) where the session lowers a scene, for `compose` and a
sync alike. `Evaluator::reset_port` and `reset_state` in the
[graph's state](../crates/core/graph/src/evaluate/state.rs);
`Session::reset_port`, recorded in history as a reset with its port;
`reset` in the protocol and `reset node.port` in a replay. Held by the
graph's tests (one port starts again and the others keep theirs; a
reset lets go of the runs that began from what it forgot, and only
those), the protocol's, and the voxel replay, which brings the camera
home and keeps the world; the scene's edit tests, which hold every byte
an edit does not name, and the protocol's, which show a node's own value
refused by `param`, changed by an edit and kept by the document, and a
declared param reaching the node that names it with the document as
written.

**What would change it.** A kind of state finer than a port (an
element, a region) that an agent must start again alone, which would
make it a write computed by a Kernel rather than a new door.

## A world is restored only when the whole world can live again

`restore` installs a snapshot's state in every state port at once, the
ports it does not name restarting from their initial sources, and with
`laws: "snapshot"` publishes its Kernels back first, all together, and
with `structure: "snapshot"` writes back the scene document the snapshot
ran in before either (the laws and the structure keep their pasts alike:
old sources and old documents by digest). Then
it proves the world by one frame. The frame runs as any does,
migrating a state whose layout the Kernels now running no longer take,
and commits nothing: when it succeeds the world is the snapshot's (the
next frame, which finds the trial's outputs in the cache, is the one
after it), and when it fails the state and the laws are put back as
they were. `laws` has no default.

**Why.** Every change in Orla is prepared, validated and proved before
it is real: a Kernel's publish, a document's Kernels together, a
composed scene, an evaluation's state. A restore is the largest change
there is, a whole world at once, and a world that cannot run a frame is
not one to return to. The trial is not committed because a snapshot is
a moment; restoring one is being there, not one frame past it. Old state
under current laws and the old world whole are different acts, so a
restore names which.

**Where.** `Evaluator::restore` in the
[graph's state](../crates/core/graph/src/evaluate/state.rs), reusing
what an evaluation replaces and puts back; `Session::restore` for the
laws; `restore` in the protocol. Held by the protocol's tests (a world
restored counts on from the snapshot; one that cannot run is refused
with its laws taken back; one restored with its laws runs as it ran)
and by the session's random life, which restores among everything else
and was checked against a trial that commits.

## A platform that lacks a primitive does the same work another way

Orla runs on WebAssembly under WASI, a machine with no threads, no
resolved links and no code made at run time. Where no thread can be
started, the work a thread would have done is done in turn: an `async`
run lands before its start returns, Kernels are prepared inline, and
`serve` answers each request before it reads the next. Where links
cannot be resolved, a file's one name is its path made whole with `.`
and `..` folded, and the sandbox, which lets no link lead out of the
directories it was given, keeps that name honest. Where nothing loads
the code a JIT writes (WASI under a plain runtime), there is no JIT,
as there is no GPU where there is no device, and asking for it says
why; where the host loads it (a page), the JIT writes WebAssembly.

**Why.** A platform is the Host's world, and the Host's primitives are
what vary; what a scene computes must not. Every fallback here is one a
native run already allows (an async run that finishes at once, a
worker that is off, a file reached by one spelling), so no document,
replay or agent sees a different world, only a slower one. A
WebAssembly backend (IR compiled to WebAssembly, the JIT's place in a
browser) is a separate step, with its own evidence.

**Where.** [`later.rs`](../crates/core/kernel/src/later.rs) and
`set_background` in the runtime,
[`store.rs`](../crates/interface/session/src/store.rs) in the session,
`stdin_into` in the protocol's transport. Held by the example replays,
run under `wasmtime` on the interpreter.

**What would change it.** Threads under WASI a runtime runs, which would
bring back the thread where it can be had, with nothing else to change.
Not `wasm32-wasip1-threads`: Orla builds for it as it is, but it rests on
the wasi-threads proposal, which wasmtime no longer runs (49 refuses
`-S threads`, and the module does not start without it), so it would be a
second runtime kept for a withdrawn proposal. The thread comes back with
WASI's shared-everything threads, once a Rust target and wasmtime both
have them. Until then the verification runs the WASI replays eight at
once, each on its one core.

## A session's documents are wherever its caller keeps them

A session reads its scene, the documents the scene uses, the files its
inputs name, and writes what its outputs and its requests write, all
through one `Store`: the disk, or the files its caller handed in and
may hand in again. The C ABI opens a session on files handed in, and a
page that runs Orla as WebAssembly, with no disk at all, does exactly
that. Where a machine has no clock, a page's, nothing is timed and
history is not dated.

**Why.** Source resolution belongs to the caller or the file loader
(AGENTS.md); the session had it by reading the disk wherever it needed
a file, which is the one thing a page does not have. A store is the
boundary that was always there, made explicit: what `use` reaches,
what a package holds, when a document changed, and where a write lands
are the same questions in memory as on disk, answered once. A file
handed in again is a file changed on disk, read at the next sync, so
hot reload is the same act in a page.

**Where.** [`store.rs`](../crates/interface/session/src/store.rs) in
the session; `orla_files`, `orla_files_put` and `orla_open_files` in
the [C ABI](../crates/interface/ffi/include/orla.h);
[`clock.rs`](../crates/core/kernel/src/clock.rs) in the runtime;
`OrlaInPage` in the web bridge. Held by the ABI's test (a scene from
paths no disk has, a Kernel put again) and the web bridge's, which runs
the text example in a headless Chrome in the page.

**What would change it.** A page that must see files change without
being told (a folder the person edits live): the page would watch and
hand them in again, and the store would not change.

## Where Orla is WebAssembly, its JIT writes WebAssembly

The JIT is the interpreter's words at a machine's speed. Where Orla is
itself WebAssembly, the machine runs WebAssembly, so that is what the
JIT writes: a Kernel's IR as a module, instruction for instruction what
the Cranelift translation does, which imports Orla's memory and function
table. The host (a page's script; node, for the tests) compiles it and
puts its functions into Orla's table, and the index it gives back is a
function pointer there, so everything around the code generator (the
five modes' drivers, the context, the helpers, integer failures, the
work budget) is the JIT's own, unchanged. What would be on a stack is
in slots the prepared Kernel owns: WebAssembly runs on one thread and a
Kernel's functions do not recurse.

**Why.** In a page, the interpreter made Orla run and nothing more: a
voxel editor took seven seconds a frame. The Kernels are not known
when Orla is built (they arrive, change and are published while it
runs), so only compiling them where they arrive makes them fast, as the
JIT does natively. A separate backend would have repeated the drivers
and the helpers; a function pointer that is a table index makes the
native JIT's host side the WebAssembly one's. No dependency is taken:
the module is written by hand, and the host is the one import.

**Where.** [`wasm/`](../crates/backends/jit/src/wasm) in the JIT, its
`wasm-host` feature, which `orla-ffi` turns on for a page; `host` in
the web bridge's [`orla.js`](../bridges/web/orla.js), which
[`wasi.mjs`](../bridges/web/wasi.mjs) uses to run Orla's WebAssembly
tests in node. Held by the JIT's own tests (the parity cases, the
feature matrix, random programs, word for word with the interpreter)
run as WebAssembly, alongside the core's, the front end's and the
interpreter's, which found two places a 32-bit `usize` saturated
first (`crates/interface/orla/tests/wasm.rs`); and by the web bridge's
test, which runs the text and voxel examples in a headless Chrome on
it.

**What would change it.** WebAssembly threads in the page, which would
let the JIT split a run across workers as it does across threads.

## In a page, waiting for the device is giving the page a turn

The browser's WebGPU answers a readback, a pipeline's validation, even
the device itself, only once the page's script has had a turn, and
Orla's evaluation waits for such answers in the middle of a frame: a
scan handed to the JIT reads what the GPU made, as does a `value(port)`,
an async node's inputs, a `read`. In a page, then, the GPU backend waits
by parking: Orla calls `orla.wait`, which the page gives as a
suspending import (JavaScript Promise Integration), and Orla is
suspended until the device's answer, or a future's waker, calls
`orla.wake`. Nothing spins while it waits; while it is suspended the
page hands it nothing, and what the person does waits for the frame to
end. The GPU backend is
the same everywhere; only how it waits differs (`orla_gpu::Wait`: block,
or park), and `orla-ffi`, which takes `unsafe`, supplies the turn.
Parked for a mapping, Orla submits nothing, and a browser may look at the
device only when something is submitted: Firefox answers a mapping then,
or otherwise a tenth of a second later. So a parking wait submits an
empty batch after it asks for a mapping, and the answer comes as soon
as the device has it; a blocking wait polls the device and needs none.
The browser is reached through wgpu's own WebGPU backend, whose glue
`wasm-bindgen` writes; its import of `orla` is the page's
`orla-host.mjs`, through an import map.

**Why.** A frame that ends only at its commit would have had to become
many frames, one for every place it waits, which is every place a
backend reads what another made: the evaluator suspended at any node,
the protocol answering "not yet", the Backend contract grown a way to
ask without waiting. Suspending the machine is the host's to offer, and
a page offers it; nothing above the GPU backend learns that it was
suspended. One GPU backend, not a second one written against the
browser, keeps its evidence one: the WGSL, the binding, the residency.

**Where.** [`wait.rs`](../crates/backends/gpu/src/wait.rs) in the GPU
backend, `Session::runtime_waiting`, `waiting` in the
[C ABI](../crates/interface/ffi/src/lib.rs), `host` and `OrlaInPage` in
the web bridge's [`orla.js`](../bridges/web/orla.js). Held by the web
bridge's test, which runs the text example's checks on the page's
WebGPU and the voxel example on it in a headless Chrome. Firefox has the
suspension from 153 and Safari from 27: every example replay played in
`play.html` on Firefox 156's WebGPU, by hand, in a window (a headless
Firefox offers no adapter); the four that read back every frame
(collapse, rubik, skirmish, stride) took thirty seconds and more before
the empty submission and as long as in Chrome after it. Safari waits
for 27 on a machine to try it.

**What would change it.** A browser that must run Orla's GPU and has no
such suspension: then the evaluation would learn to stop and go on, and
this would be one way of waiting among two.

## What playing a scene means is decided once

Every interface that plays a scene shares one play
(`orla_session::play`): the frame (what changed is read, a replay's
frame taken at its time, the scene evaluated, the picture and the sound
named), what F5, Tab, F10 and Escape mean, how a pad is a keyboard, how
the pointer is found in the picture, what the line of frame data says,
and what is said on the way. A window, a page, anything that shows a
scene, is its devices only: it hands the play its person's hands and
its clock, shows the picture on its own device and plays the sound on
its own, and tells the play where the picture landed.

What every interface asks of a session is the session's to answer, not
each interface's: which output is the picture and which the sound
(`Session::picture_of`, `sound_of`), a Field or a picture read back
from the device (`Session::field`, `picture`; no interface reaches
into the runtime to change it), the backend that runs when none is
named (`best`: the GPU, else the JIT, else the interpreter, and the
ones passed over said), the session's own flags for every command
(`SessionOptions`, `graph` too), and what checking a document finds
(`check_document`, through the store). How each drives its frames
stays its own: a replay at its written times, a window at its clock,
`advance` at a rate until a value says stop, `render` at a fixed rate;
what a frame is, `Session::evaluate`, is one.

**Why.** ARCHITECTURE-PLAN §34: interfaces are many, and "all of them
operate on the same underlying Runtime". A second Player written for a
page would have been the first one again, and the two would have
drifted apart. A display, keys and a sound card are the Host's IO
(§6); a page is another Host, not another product. The session is
already what "every interface drives"; playing a scene is part of it.

**Where.** [`play`](../crates/interface/session/src/play/mod.rs), with
its [pad](../crates/interface/session/src/play/pad.rs) and
[line of frame data](../crates/interface/session/src/play/hud.rs); the
window, sound card and pads of [the Player](../crates/interface/player/src/main.rs).
A page's Player is its devices over the same play, through the C ABI
(`orla_play_*`, [play.rs](../crates/interface/ffi/src/play.rs)):
[player.js](../bridges/web/player.js) hands it the page's keys, pointer,
wheel and pads and its clock (`orla.now`), and shows, plays and watches
on the page. Held by [the play's tests](../crates/interface/session/tests/play.rs),
headless: the keys, the preview, the pointer, the sound output, a
failure said once, a replay played to its end; by the ABI's test, a
replay played through `orla_play_*`; and by the web bridge's test, a
replay played in `play.html` on the page's GPU and JIT, and F5 and Tab
pressed there.


## A replay is a test; a piece is made of laws, structure and state

A replay is input played into a scene over time, and what it expects
of the scene: a test. It lives under its example's `tests/`, and every
one there is a test of what that example promises. Played headless
(`orla replay`), it holds the scene: frame after frame as fast as they
compute, which is what the verification runs. Played by the play the
Player and a page share (`orla-player --replay`, `play.html?replay=`),
it holds the interface as well: the keys and pads mapped, the pointer
found in the picture, the picture shown, a page's device waited for;
and it can be watched, a failure seen where it happens, an interaction
shown to someone. The window is handed over when it ends, for looking
around, not as a way back to a world.

A piece made with a scene is kept as what makes it, never as the
clicks that made it: its laws as Kernels (`publish`), its structure as
the scene document (`compose` and its `edits`), and what it came to as
state: a port's Field (`read`, `write`, `save`) or the whole world (a
`snapshot`, and `restore`).

**Why.** A replay of clicks is a test of the editor clicked: move its
bar, its camera or its picking, and the piece is gone, while the test
fails as a test should. The voxel monolith, built by hand as a replay,
was that: kept among the tests, it cost more than any of them (31 s of
the WASI replays' 79) and held nothing a test holds. Headless and
played are one replay and one play, not two: headless is the faster
and checks the scene; only played does the interface run, and a fault
that is the interface's (Firefox answering a mapping a tenth of a
second late) is found there and nowhere else.

**Where.** [The replay module](../crates/interface/session/src/replay.rs)
reads and runs a replay; [`play`](../crates/interface/session/src/play/mod.rs)
plays one. Held by the example replays under the verification, the
Player's test that a replay played in a window holds as it does
headless, the ABI's through `orla_play_*`, and the web bridge's in
`play.html` on the page's GPU and JIT.

**What would change it.** A piece that is its making, a recorded
performance kept for its own sake, which would be data a scene reads,
not a replay among the tests.
