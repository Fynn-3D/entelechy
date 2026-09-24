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
takes its allowances from the command line that started it.

**Where.** [`writes.rs`](../crates/interface/session/src/writes.rs) in the
session; `to` and `else` in the document
([parser](../crates/frontend/language/src/parser.rs),
[lowering](../crates/frontend/scene/src/lib.rs)); a fallback in use
substitutes for its input when the evaluation starts
([evaluator](../crates/core/graph/src/evaluate.rs)). Held by
[the session's write tests](../crates/interface/session/tests/writes.rs),
[the lowering's](../crates/frontend/scene/src/lib.rs) and
[`examples/save`](../examples/save/README.md).

**What would change it.** An effect whose target the Kernel must choose
while running (a file named by a Field, a network request), which would
make the target part of the data as well; a second effect beside files,
which would get its own allowance; or an effect that must happen before a
frame commits, which this model refuses on purpose.

## What a served scene reaches is what the person running it gave

A session an agent drives, `orla serve` or the Player's `--serve`, holds
the scene to what the person who started it allowed, and nothing that
arrives over the wire widens it. An agent's `save` and a scene's `to`
land only where `--allow file.write` says. An input reads a file only
beside the documents the scene reads (the root, the graphs it places,
what it uses), inside a package, or where `--allow file.read` says. A
browser's page opens the socket only from this machine, from a file, or
from an origin `--allow-origin` names, and an HTTP request that is no
WebSocket upgrade is answered and closed. A Kernel or a scene an agent
sends is bounded before it costs anything: a domain is charged against
the Host's budget before it is allocated, an `[each]` and a `times`
before they are written out, and a rejected scene never reaches the
disk.

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
reads, origins and compose.

**What would change it.** A second person on the same session, which
would want who asked, not only what; a network listener beyond this
machine, which would want the socket itself authenticated.
