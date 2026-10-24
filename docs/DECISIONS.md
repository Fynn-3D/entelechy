# Decisions

**Answers:** why Entelechy is this way and not another, where the answer is
not visible in the code.

Each entry is a ruling made inside the architecture documents'
boundaries, and holds until it is replaced by name: the choice, why, and
where the code holds it. How a ruling is carried out is the code's to
say, beside the code.

## The Graph computes structure, never values

A scene's only computation is structural: how large a node's output is
(`domain`) and how many times a node is placed (`times`). A node's value
is written, a literal or a graph parameter by name. A domain reads values
(a parameter, the window, a number a Kernel wrote with `value(port)`) and
yields only a count or a size, exact or failing, never wrapping. A graph
says what its parameters must hold with `require name: a <= b;`, two
counts compared, checked as the document loads and before every
evaluation; a placed graph's requirements come with it.

**Why.** A scene that computes values is a second language beside
`.ent`: untyped, untraced, run on the Host rather than a backend. What
breaks a preset is a relation between its parameters that no single
Kernel sees whole; it belongs to the graph that states them, and a
verdict over counts is still structure.

**Where.** The literal reader in [`scene`](../crates/frontend/scene/src/lib.rs);
`Size` and `Requirement` in [`domain.rs`](../crates/core/graph/src/domain.rs).

## The Host bounds work and memory

Each element of a Kernel runs at most its budget of loop iterations
(2^24 unless it declares `budget N;`); one run, all elements together,
at most the Host's bound (`--work`). What a run holds in Host memory (a
scatter's pairs, its sorted copies) counts against the Host's budget as a
Field does. Past any bound the run fails, named; nothing commits. Both
work bounds count iterations, never time; where they are counted is
written into the IR once, and a backend implements a count, never where
it goes.

**Why.** A Kernel that never finishes, or that takes the machine's
memory by asking, is a failure no validation catches. The Kernel owns
its budget because it knows its loops; the Host owns the run's because
it owns the machine. Counting iterations makes a failure the same on
every backend.

**Where.** [`account`](../crates/core/ir/src/account.rs),
[`Host::work`](../crates/core/host/src/lib.rs),
[`Scope::room`](../crates/core/abi/src/lib.rs); tests
`an_element_past_its_budget_fails_alike_on_every_backend` and
`a_run_past_the_hosts_work_fails_alike_on_every_backend`.

## A function never reaches itself; its place in the document is free

Functions call one another in any written order, within a document and
across what it uses; a chain of calls that comes back to a function is
refused, named whole (`a -> b -> a`). Repetition is a `loop`.

**Why.** A Kernel has no stack: WGSL forbids recursion and an element's
work is bounded by its loops alone. Refusing the cycle protects that
without making a document's order a thing to arrange.

**Where.** `refuse_cycle` in the compiler's
[`lower`](../crates/frontend/compiler/src/lower/mod.rs).

## One expression has one answer on every backend

Where a machine leaves a result to its hardware, the language decides it
once:

- Float `max` and `min` are IEEE 754-2019 maximumNumber and
  minimumNumber: a NaN gives way, and `max(-0.0, 0.0)` is `0.0` either
  way round.
- An integer `sum` of a scatter or prefix is each element's whole total,
  failing only when that total does not fit, at the first such element.
- The clock crosses into a Kernel whole, a `Double` of seconds:
  `clock()` reads it so, `time()` rounded once to a `Float` by the
  context.

**Why.** A semantics a backend may choose is not one: a machine's `max`
gave `inf` on one backend and `-inf` on another, and a `Float` clock
stepped in jumps a session a month old. A result decided by its inputs,
not by the machine or the order it adds, is the same everywhere.

**Where.** [`entelechy_rules`](../crates/core/rules/src/lib.rs) (`max_f32`,
`Sums`), the GPU's [prelude](../crates/backends/gpu/src/wgsl/prelude.rs),
`Context::float_time` in the [ABI](../crates/core/abi/src/lib.rs); held by
the [conformance matrix](../crates/backends/conformance/src/lib.rs) and the
JIT's [random programs](../crates/backends/jit/tests/fuzz.rs).

## A group's lanes meet at exchanges

A `group N` Kernel is a map run `N` elements at a time, and
`exchange(v, k)` hands each lane the value lane `k` holds at the same
point: no shared array, barrier or atomic is written. Every lane reaches
each exchange alike, which the IR checks; a failure is the group's,
named at the lowest lane that failed in the first stretch it failed in.
The owner chose this form.

**Why.** An FFT, a block sum or a sort network needs elements that see
one another inside a pass. Shared memory with barriers lets a lane race
or wait forever, which no reference can hold to one answer; an exchange
is a value, so a race cannot be written and every backend computes the
same words.

**Where.** [`group`](../crates/core/ir/src/group.rs) and
[`lanes`](../crates/core/ir/src/lanes.rs) in the IR; the group mode of
the [conformance matrix](../crates/backends/conformance/src/lib.rs).

## A frame runs what something reads; a check says what running would

A frame runs the nodes an output, a state, or a pin reads, and what they
read in turn; a node nothing reads does not run, and `entelechy check`
notes it. A `compose` trial runs every node. `entelechy check` holds
every node to what the types alone decide (each input's shape and
layout, each domain's shape), in the words the evaluation would use;
what the data decides is the evaluation's.

**Why.** A Kernel is pure: a node nothing reads changes nothing anyone
sees, so a frame costs what it shows. A fault the types decide is found
before anything runs, on no particular backend, by one function the
binding and the check share.

**Where.** `Graph::read_through` ([graph](../crates/core/graph/src/lib.rs)),
`type_fault` ([bind](../crates/core/graph/src/evaluate/bind.rs)); held by
[the unread tests](../crates/interface/session/tests/unread.rs) and
[the binding tests](../crates/interface/session/tests/binds.rs).

## The language grows only by the owner's word

A form, a keyword, a builtin or a shorthand is added to `.ent` or `.esn`
only when the owner asks for it or agrees to it in words. A contributor,
an agent above all, who finds writing hard meets it with what the
language already has: a function, a package, or an example that shows
the way. A proposal to grow the language is put to the owner, answering:

1. It lowers to the IR there is: no instruction, backend behavior,
   budget rule or runtime concept is new.
2. How often sources in and beside the repository write what it stands
   for, counted, not felt while writing one demo.
3. Why a function, a package or the forms there are cannot write it, or
   write it only wrongly.
4. It is a word or a builtin everywhere it may appear, never one that
   means something in one place alone.

**Why.** Every form is one more thing each creator, an AI above all, must
learn, and one more thing every later rule must hold for. Friction met
one demo at a time grows a language by the sum of local pains, which no
one decided. A function or a package is a condition features come from;
a shorthand for one pattern is not.

**Where.** [`surface.txt`](../crates/frontend/compiler/tests/surface.txt)
holds every builtin and form, and [`surface.rs`](../crates/frontend/compiler/tests/surface.rs)
fails when it grows; the scene syntax is held by
[the scene documents' surface](../crates/frontend/scene/tests/surface/).

## A public contract grows; it breaks only with its version

Six things are written against by programs Entelechy does not see. Each
may grow; what was valid against it stays valid, or its version changes.

| Contract | Held by |
| --- | --- |
| Protocol operations | the table in [the protocol module](../crates/interface/protocol/src/lib.rs), held to `OPERATIONS` and the server answering each |
| C ABI | [`header.rs`](../crates/interface/ffi/tests/header.rs) |
| Capabilities | [`contract.rs`](../crates/core/abi/tests/contract.rs), pinned to `ABI_VERSION` |
| `.field` | [`field_file.rs`](../crates/interface/formats/tests/field_file.rs) |
| Language surface | [`surface.txt`](../crates/frontend/compiler/tests/surface.txt) at `LANGUAGE_VERSION` |
| Scene documents | [`surface/`](../crates/frontend/scene/tests/surface/), each beside the Graph it lowers to |

**Why.** A contract changed by accident breaks a reader who is not
present. A test that fails on any difference makes every change a
decision, and the version says which kind. A new form is recorded with
`ENTELECHY_BLESS=1`; dropping or tightening one is refused until the
version is raised.

## Sources are the loader's; names are the language's

The loader finds documents: it resolves each `use` to files and
packages, refuses cycles, and hands the language what it reached. The
language decides every name, and a declaration carries where it was
written, so a fault is placed by looking it up.

- **A document sees what it uses itself.** Its names are its own and
  those of the documents its `use` items name; a name it reaches only
  through another's `use` is refused, with the `use` that brings it.
- **A name is one package.** `voxel@1` is `voxel.entpkg`; a scene
  reaches one package of a name. Where two packages a document uses
  declare one name, it writes `package::name`; the language resolves it
  before lowering, so no later layer sees a qualified name. The owner
  chose this form.

**Why.** Names split between the session and the language were merged
twice and placed in the wrong file at the seams; one owner leaves
nothing to take apart. A name seen through another document is a
dependency no line states. A short word is one two packages want alike;
the package's name already says whose it is.

**Where.** [`scope.rs`](../crates/frontend/language/src/scope.rs) and
[`sources.rs`](../crates/interface/session/src/sources.rs); held by
[the `sources` tests](../crates/interface/session/tests/sources.rs).

## Entelechy describes itself, from the tables it enforces

The language's types, builtins and forms, the scene's forms, the
capabilities, a package's parts, the protocol and its wire, the replay
lines and the command line are one reference, rendered from the tables
the implementation executes. `help` answers it; `docs/REFERENCE.md` is
the same text. Whole programs are the examples.

**Why.** A description rendered from what runs cannot drift from it. An
agent learns fastest from one dense reference and whole programs;
anything built beyond that to teach it costs more than it saves.

**Where.** [`help`](../crates/interface/protocol/src/help.rs); held by
[the signatures compiled](../crates/frontend/compiler/tests/help_signatures.rs)
and [the forms parsed](../crates/frontend/compiler/tests/help_examples.rs).

## A rule that had two owners is held to one

A rule found with a second owner (a syntax parsed twice, a second way to
the disk, an interface asking the session's question) is left with one,
and held there: by the compiler where visibility can say it, by a source
scan where it cannot. A scan row names the act a copy would do, its
owner, and the commit that gave it one; a row is added only after a rule
had a second owner, and is rewritten, never widened, when it fails on
legitimate code.

**Why.** A rule with two owners drifts until they disagree, and tests
hold behavior, not ownership. A guard per broken rule is few and
precise; a guard per rule is patterns readers learn to silence.

**Where.** [`owners.rs`](../crates/interface/entelechy/tests/owners.rs)
and [`layers.rs`](../crates/interface/entelechy/tests/layers.rs).

## A Kernel's effects are data the Host carries out after a commit

A Kernel never writes a file. A scene declares an output also written
to a file (`to "saves/state.field"`, or `to files "dir/"` for a list of
writes, removals and renames), and the session carries it out once the
evaluation that changed it has committed, whole, and only where the
person running the scene allowed (`--allow file.write`, `file.delete`).

**Why.** An effect inside a Kernel breaks what a Kernel is: the same on
every backend, committing nothing when it fails, replaying alike. As
data, the effect is typed and traced, and is carried out where nothing
can undo it. The allowance is the person's, never claimed by a Kernel or
an agent.

**Where.** [`writes.rs`](../crates/interface/session/src/writes.rs);
held by [the write tests](../crates/interface/session/tests/writes.rs).

## A world may write its own laws

A scene may write the documents its Kernels are published from, through
an output `to files` into a directory the runner allowed; the session
compiles, validates and swaps what it wrote as it does any edit, and a
law the world gets wrong is refused while the last runs on. The history
names such a law the world's own. A replay may allow its scene to write,
into a store laid over the disk that never reaches it. The owner chose
this form.

**Why.** A world whose laws only an outside author changes is made, not
alive. A law is a document, so writing one is writing a file: the safe
path the Host already has, checked by its allowance and swapped by hot
reload. A world writes text, not IR, held to the language as every
author's law is.

**Where.** `Origin::Output` in the [history](../crates/interface/session/src/history.rs),
`Kept` in the [store](../crates/interface/session/src/store.rs); held by
[the genesis tests](../crates/interface/session/tests/genesis.rs) and
[the genesis example](../examples/genesis/README.md).

## What a served scene reaches is what the person running it gave

A session an agent drives holds the scene to what the person who
started it allowed; nothing over the wire widens it. Reads are held to
the scene's directories as it first loads, the package directories and
`--allow file.read`; writes to `--allow file.write`; a socket answers
this machine or an origin `--allow-origin` names; a Kernel is prepared
only with the grants it requires; and what an agent sends is charged
against the Host's budget before it costs anything.

**Why.** A file an agent can read it can send away, and any page the
person visits can reach a socket on their machine.

**Where.** The session's [inputs](../crates/interface/session/src/inputs/mod.rs)
and [writes](../crates/interface/session/src/writes.rs), the protocol's
[transport](../crates/interface/protocol/src/transport.rs); held by
[the protocol tests](../crates/interface/entelechy/tests/protocol.rs).

## What a scene reads of the Host's world is an input or a node

The Host hands a scene what it reads from outside through the forms it
already has, on the sync that already watches files:

- **Formats.** PNG, PPM, PGM, WAV and MIDI are read live; other bytes go
  to a Kernel as `Rows<Int>` (`bytes`); glTF, fonts, SVG and lossy sound
  are converted outside the runtime by `entelechy import`.
- **The language's view of documents.** `checks "dir/"` gives the
  problems, `outline "dir/"` the definitions, a line each, as text.
- **A directory.** `files "dir/"` as an input gives every file's bytes;
  as a node, the listing git would track, and the bytes of the files the
  graph names in `paths`.
- **A world larger than memory.** `node n: Rows<E> fields "dir/"` gives
  the elements of the Field files the graph names, read off the frames
  where the session prepares in the background, at once in a replay.

Each was chosen by the owner, and each lowers to what a scene already
has: an input, or externals filled a frame late as a font node's are.

**Why.** A scene that edits, browses or streams needs the Host's files,
but no new lifecycle: what the graph asks for comes from the graph, as
it runs. A format is read live only when it decodes to a numerical
Field without a dependency outside the workspace; a closed set keeps the
formats crate from collecting every format. Text is what every
interface already says a problem in, and what a Kernel reads as it
reads any text. Only what a document declares is given, so the compiler
stays out from under the runtime.

**Where.** [`entelechy-formats`](../crates/interface/formats/src/lib.rs);
the session's [inputs](../crates/interface/session/src/inputs/mod.rs); held
by the session's checks, outline, files, files node and fields node
tests, and the scene surface's documents of each form.

## Reality changes atomically; history records why

A session's position is `(revision, frame)`: the revision moves when the
world changes other than by running, the frame counts the evaluations
committed within it. Every change is recorded with its origin, sources
and documents are kept once each by digest, and refused attempts are
recorded apart, moving nothing. A snapshot takes committed state whole,
with its laws; `restore` installs it, with `laws` and `structure` said,
proves the world by one frame, and puts everything back if it cannot
live.

**Why.** An agent changing a live world needs to know how it came to be
and what was refused. Only committed state is taken, because the cache
may hold what a failed evaluation never committed. A restore is the
largest change, so it is proved before it is real.

**Where.** [`history.rs`](../crates/interface/session/src/history.rs),
`Evaluator::restore` in the [graph's state](../crates/core/graph/src/evaluate/state.rs);
held by the protocol's and the history's tests.

## A live world is changed through three doors, each at its own grain

- **Structure**, a statement at a time: `compose` edits the scene's
  documents (the scene, and the Kernel, graph and replay documents under
  its directory that it uses, places or tests), several as one, all
  written or none, each checked whole before any is written.
- **State**, a committed port at a time: `write`, `reset`, `restore`.
- **Parameters** the document declares.

Nothing else reaches in.

**Why.** Each door is where the thing it changes is kept whole. A node
value set from outside is structure the document does not say, lost at
the next read. The document is also a person's: an agent names what it
changes and every other byte stays. A port is the smallest part of state
that restarts without inventing a state no evaluation committed.

**Where.** [`edit`](../crates/frontend/scene/src/edit.rs) in the scene,
`Session::write_documents`, and the [graph's state](../crates/core/graph/src/evaluate/state.rs);
held by the edit, graph and protocol tests.

## What playing a scene means is decided once

Every interface that plays a scene shares one play: the frame, what F5,
F6 and F10 mean (every other key is the scene's), how a pad is a
keyboard, how the pointer is found in the picture, and what is said. An
interface is its devices only. What every interface asks of a session
(which output is the picture, the best backend, every command's flags,
what a check finds) the session answers.

**Why.** ARCHITECTURE-PLAN §34: interfaces are many, and all of them
operate on the same underlying Runtime. A display, keys and a sound card
are the Host's IO; a page is another Host, not another product.

**Where.** [`play`](../crates/interface/session/src/play/mod.rs); held by
[the play's tests](../crates/interface/session/tests/play.rs) and the web
bridge's.

## A replay is a test; the framework tests itself

A replay is input played into a scene over time and what it expects,
under its example's or package's `tests/`; played headless it holds the
scene, played in a window the interface too. A piece made with a scene
is kept as its laws, structure and state, never as a replay of clicks.
Rust tests hold the framework with scenes written in the test, and never
read `examples/` or `packages/`; a demo or a package holds itself by its
replays, on every backend, nothing fused, rounded apart.

**Why.** A replay of clicks tests the editor clicked. A demo's author
needs no Rust to validate it, and a framework tied to its demos slows
with each and collects their exceptions; a package carried alone takes
its tests with it.

**Where.** [The replay module](../crates/interface/session/src/replay.rs),
the recipe's `replays` step in [PROCESS](../PROCESS.md), `tested` in
[the packages' check](../crates/interface/xtask/src/packages.rs).

## A platform is the Host's world; what a scene computes is not

- Where no thread can start (WASI), work is done in turn; where links
  cannot be resolved, a file's name is its path made whole.
- A session's documents are wherever its caller keeps them, through one
  `Store`: the disk, or files the caller handed in, as a page does.
- Where Entelechy is WebAssembly, its JIT writes WebAssembly, and the
  host compiles it into Entelechy's function table.
- In a page, waiting for the device is giving the page a turn: the GPU
  backend parks on a suspending import until the browser answers.

**Why.** A platform's primitives vary; each fallback is one a native run
already allows, so no document, replay or agent sees a different world.
Source resolution belongs to the caller. Suspension is the host's to
offer, so nothing above the GPU backend learns it was suspended.

**Where.** [`later.rs`](../crates/core/kernel/src/later.rs),
[`store.rs`](../crates/interface/session/src/store.rs), the JIT's
[`wasm/`](../crates/backends/jit/src/wasm), the GPU's
[`wait.rs`](../crates/backends/gpu/src/wait.rs); held by the WebAssembly
tests and the web bridge's, in a headless Chrome.

## The rasterizer is a package; the native one is its fast path

A scene places `packages/raster.entpkg` to turn triangles into a
surface, and binds a preset's native operation (`native.rasterize@1` to
`@3`) in its place where it wants speed. Both keep one rule: a pixel is
covered where its center is inside the triangle's box and on or inside
all three edges, holding the nearest fragment, the earlier on a tie. On
the interpreter and the JIT the native path is exact to the rule; on a
GPU it is exact inside drawn triangles, and the device's rasterizer may
differ at edges.

**Why.** As Kernels, rasterization is typed, traced, hot-reloaded and
changeable by an author; a native operation is a package's fast path
held to a Kernel definition, never a second definition. A device may
round one expression differently in two Kernels, so a decision two
Kernels share is made in one; `entelechy replay --round-apart` finds one
made twice.

**Where.** The package and its presets; `entelechy_rules::raster`; the
GPU's [parity tests](../crates/backends/gpu/tests/parity.rs); the mesh
example's edge, native, cutout and canvas replays.

## Where a run goes is the backend's; what it computes is not

Beside a device, a run goes to the JIT when its elements times its
Kernel's typical work are at most `--small` (or when one of the device's
groups holds all its elements), and to the device otherwise; a run that
reads what the device just made stays there. A Kernel the device's own
compiler cannot build runs on the JIT from then on; a shader the backend
emitted that the device refuses is the backend's fault and still refuses
the document. `entelechy check` says when a Kernel is large for a
device's compiler, by the words it carries through its branches.

**Why.** A device's dispatch costs about the same whatever it does, so a
device earns its cost over many elements and nothing over few. The
choice reads the run and the Kernel, never a clock, so a replay places
every run alike, and the JIT computes what the reference does.
Placement is a backend's to decide and the document's to read about,
never to fail on.

**Where.** `Runtime::execute` in [`entelechy-kernel`](../crates/core/kernel/src/lib.rs),
the GPU's [`declines`](../crates/backends/gpu/src/lib.rs),
[`size.rs`](../crates/core/ir/src/size.rs); held by
[`routing.rs`](../crates/interface/session/tests/routing.rs),
[the unbuilt test](../crates/interface/session/tests/unbuilt.rs) and
[the large test](../crates/interface/session/tests/large.rs).

## A change prepares off the frames; nothing invalid publishes

Preparing a Kernel validates it and builds nothing: a device builds it
when it is first needed, or as it publishes where a node will surely run
there. A changed document is read and compiled off the frames while the
running Kernels keep them, and publishes whole on the sync after. Where
a session shows while building (served, or asked), a Kernel that only
reaches what is shown runs on the processor until the device has built
it; tests, renders and replays wait for every Kernel.

**Why.** Hot reload must not stall a frame, and a device's compiler
takes seconds on a whole-screen Kernel. Moving work off the frames
changes when a change publishes, never what publishes. Showing from the
processor is held to what only shows, because rounding then depends on
how long a compiler takes, which nothing kept may.

**Where.** The GPU's [`Lazy`](../crates/backends/gpu/src/prepare.rs),
`Scene::sync_libraries` ([scene](../crates/interface/session/src/scene.rs)),
`Graph::shown_only`; held by the session's lazy, compile-ahead and
shown-while-building tests.

## Work is spared only where the answer is the same

Every saving is keyed by what the work depends on, so it never changes
what a frame computes:

- A node whose cache key (generation, parameters, inputs' epochs,
  domain, clock, input) is unchanged is not run; a replay played again
  takes each frame's remembered outputs under the same key.
- Outputs nothing reads again are let go: those made once, once made,
  and those new frame after frame, once read; what stays is their key
  and epoch, so a reader made again finds them made again alike.
- A Field is carried page by page: what a run costs is the pages it
  changes, while the Field keeps its declared extent and the budget
  counts it whole.
- A producer read at its reader's element, or at one shared offset,
  fuses into the reader.
- A window's frames overlap the device's: a frame commits ahead and is
  confirmed by the next, and a failure takes both back.
- A frame's device time is told by the device as it finishes, never by
  waiting for it.

**Why.** An optimization that changes the answer is a second semantics.
The key already names everything an output depends on, so held outputs
under an equal key are the outputs, and outputs made again are trusted
to repeat only where a backend's conformance says so. A reading that
waits for the device changes what it reads.

**Where.** The evaluator's [recall](../crates/core/graph/src/evaluate/recall.rs),
[ledger](../crates/core/graph/src/evaluate/ledger.rs) and
[frames](../crates/core/graph/src/evaluate/frames.rs), the Host's
[pages](../crates/core/host/src/lib.rs), the IR's
[fusion](../crates/core/ir/src/fuse.rs); held by the session's recall,
memory, composition, offset and overlap tests.

## A package speaks in functions, Kernels, graph presets and data

A package's capability is words a creator composes, never a feature
built for one use, and its words come in four layers, each composed
where it belongs:

1. **A function** composes inside a Kernel, on one element. Whatever can
   be said of one element alone is said as a function first.
2. **A Kernel** is a node of the Graph only where its work cannot be one
   element's: it reads its neighbors, the last frame, another size, or
   is a pass a scene places on its own. Its body calls the package's
   functions.
3. **A graph preset** (`graphs/*.esn`, placed as a subgraph) is a way
   the package recommends to compose its Kernels: a recipe, not an
   interface. A scene replaces any part of it with its own node.
4. **Data** is a look or a setting as rows (`Grade`, `Film`, `Dof`),
   made by a node and carried by the Graph: what a story, a time or a
   mask changes. A Kernel's parameters are the defaults a node starts
   from.

A package holds no person's input (that is `controls`'), no switch
between two looks (a look turned into another is data blended), and no
content one program alone holds.

**Why.** A Graph is a program because its nodes are words and its edges
the relations between them. A Kernel that holds a whole effect hides
those relations where no scene can take one part and leave another.
Words at every layer keep composition open: a scene builds the effect
it wants, and the package grows by words, not by features.

**Where.** [packages](../packages/README.md) holds the packages to their
names, layers and tests (`cargo xtask packages`); the light package's
[words test](../packages/light.entpkg/tests/words.esn).
