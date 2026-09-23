# Implementation evidence

**Answers:** where the code is, and which test establishes each contract.

Every line here points at code or at a test. Follow the link before repeating
what it says; the test is the claim, this is the index to it. Syntax lives in
[DOCUMENTS](docs/DOCUMENTS.md), the wire in [PROTOCOL](docs/PROTOCOL.md),
commands in [USAGE](docs/USAGE.md); this file keeps what they cannot: where a
contract is enforced, and the constraints that are not visible from the code.

## The four proofs

| Proof | Where it is exercised |
| --- | --- |
| `.orla` compiles | [compiler tests](crates/compiler/tests/compile.rs) |
| Kernel calls Host only through ABI | [scope tests](crates/abi/src/lib.rs), [denied execution](crates/cpu/tests/execute.rs), [validator capability checks](crates/ir/src/validate.rs) |
| Kernel hot-swaps without restart | [hot swap tests](crates/runtime/tests/hot_swap.rs), [session tests](crates/cli/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/scene/tests/composition.rs), [example scene](examples/grayscale/scene.osn) |

## Layers

| Layer | Implementation | Evidence |
| --- | --- | --- |
| Host Core | [host](crates/host/src/lib.rs), [files](crates/host/src/file.rs) | handles, budgets, bounds |
| Capability ABI | [abi](crates/abi/src/lib.rs), [names](crates/abi/src/names.rs) | grants checked before every Host call |
| Orla IR | [ir](crates/ir/src/lib.rs), [validation](crates/ir/src/validate.rs), [display](crates/ir/src/display.rs), [fusion](crates/ir/src/fuse.rs), [optimizer](crates/ir/src/optimize.rs) | typing, SSA order, capability coverage; [fusion bit-identity](crates/ir/tests/fusion.rs) |
| Kernel language | [language](crates/language/src/lib.rs), [parser](crates/language/src/parser.rs), [lexer](crates/language/src/lexer.rs) | syntax, precedence, positioned errors, near misses |
| Compiler | [compiler](crates/compiler/src/lib.rs), [lowering](crates/compiler/src/lower/), [help](crates/compiler/src/help.rs) | [compile tests](crates/compiler/tests/compile.rs) |
| Shared rules | [rules](crates/rules/src/lib.rs), [the `Backend` contract](crates/rules/src/backend.rs), [reduce tree](crates/rules/src/reduce.rs), [matrix formulas](crates/rules/src/matrix.rs), [rasterizer contract](crates/rules/src/raster.rs) | every backend's parity test runs by them |
| Reference backend | [cpu](crates/cpu/src/lib.rs), [word code](crates/cpu/src/eval/) | [execution tests](crates/cpu/tests/execute.rs), [optimizer tests](crates/cpu/tests/optimizer.rs) |
| JIT backend | [jit](crates/jit/src/lib.rs), [Cranelift emission](crates/jit/src/codegen.rs) | [parity](crates/jit/tests/parity.rs), [matrix](crates/jit/tests/matrix.rs), every replay on `jit` |
| GPU backend | [gpu](crates/gpu/src/lib.rs), [WGSL emission](crates/gpu/src/wgsl.rs), [rasterizer](crates/gpu/src/raster.rs), [presenter](crates/gpu/src/present.rs) | [parity](crates/gpu/tests/parity.rs), [matrix](crates/gpu/tests/matrix.rs); both skip without an adapter |
| Kernel Runtime | [runtime](crates/runtime/src/lib.rs), [off-frame runs](crates/runtime/src/later.rs); takes validated IR and the backend it is given | [hot swap](crates/runtime/tests/hot_swap.rs), [async](crates/runtime/tests/later.rs), [mixed clocks](crates/runtime/tests/mixed.rs) |
| Scene documents | [scene](crates/scene/src/lib.rs), [domain expressions](crates/scene/src/domain.rs) | [lowering tests](crates/scene/src/lib.rs) |
| Graph Runtime | [graph](crates/graph/src/lib.rs), [evaluator](crates/graph/src/evaluate.rs), [domain](crates/graph/src/domain.rs) | [composition tests](crates/scene/tests/composition.rs), written as documents |
| Session | [session](crates/session/src/lib.rs), [bench](crates/session/src/bench.rs), [file formats](crates/session/src/) | [session tests](crates/cli/tests/session.rs) |
| Creation protocol | [protocol](crates/session/src/protocol.rs), [WebSocket](crates/session/src/websocket.rs), [contract](docs/PROTOCOL.md) | [client test](crates/cli/tests/protocol.rs); [an agent's record](examples/scanline/agent.jsonl) replayed |
| Command line | [cli](crates/cli/src/main.rs), [replay](crates/cli/src/replay.rs), [render](crates/cli/src/render.rs), [packages](crates/cli/src/package.rs), [scaffold](crates/cli/src/scaffold.rs) | every example replay on three backends |
| C ABI | [ffi](crates/ffi/src/lib.rs), [header](crates/ffi/include/orla.h) | [ABI tests](crates/ffi/tests/abi.rs), [C example](bridges/c/hello.c) |
| Player | [window loop](crates/player/src/main.rs), [audio](crates/player/src/audio.rs), [gamepad](crates/player/src/gamepad.rs) | presentation and blit tests |
| Import and export | [import](crates/import/src/), [export](crates/export/src/) | [MIDI tests](crates/import/tests/midi.rs), [round trip](crates/export/tests/round_trip.rs) |
| Bridges | [bridges](bridges/), each in its program's own language | [bridge tests](crates/cli/tests/bridges.rs), run when the program is present |
| Feature matrix | [matrix](crates/matrix/src/lib.rs) | [on the JIT](crates/jit/tests/matrix.rs) bit for bit, [on the GPU](crates/gpu/tests/matrix.rs) ints exact and floats to rounding |

## Libraries, not primitives

What these packages do needed no engine feature; keep it that way.

| Library | Built from | Evidence |
| --- | --- | --- |
| [Stream compaction](packages/compact.orlapkg/kernels/compact.orla): a rank is `prefix with sum`, the count a reduce, the placement a scatter whose outside addresses drop | three execution modes and `value(port)` | [culled replay](examples/culled/tests/culled.replay), the same picture culled and not |
| [Radix sort](packages/sort.orlapkg/kernels/sort.orla) | prefix and scatter | [sort replay](examples/sort/tests/sort.replay) |
| [Mip levels](packages/picture.orlapkg/graphs/mip.osn): a sub-graph placed `levels` times with `carry`, the level read off a rasterized surface's neighbouring `uv` | `times`, `carry`, the rasterizer's `uv` and `id` | [mipmap replay](examples/mipmap/tests/ground.replay) |
| [Text](packages/text.orlapkg/kernels/text.orla): labels are `Rows<Glyph>`, the font a `Grid<Float>` atlas baked by a Kernel | shared functions, assets | [text replay](examples/text/tests/text.replay) |
| [Skinning](packages/skin.orlapkg/kernels/skin.orla): a rig's rows under the [rig contract](docs/RIGS.md) | maps over rows | [rig tests](crates/cli/tests/rigs.rs), [skin](examples/skin/tests/skin.replay) and [crowd](examples/crowd/tests/crowd.replay) replays |

## Contracts

### Resource and residency

- One resource, following [the Typed Field decision](Orla%20Typed%20Field%20Architecture.md):
  a Field of `Rows`, `Grid` or `Volume`, one 32-bit word per scalar
  ([host](crates/host/src/lib.rs)). Port types are structural; `Image` is
  the compiler's alias for `Grid<Vec4>`. Bindings are checked by shape and
  layout, never by spelling ([evaluator](crates/graph/src/evaluate.rs)).
  Every backend carries elements as words, so integers and booleans are
  exact everywhere.
- The Host owns every Field; a slot may hold its words on a device as an
  object only the backend understands. The GPU binds resident inputs,
  uploads Host inputs once, and leaves outputs resident, so committed state
  ping-pongs on the device. Words return only at observation boundaries
  through [`Runtime::image`](crates/runtime/src/lib.rs) — export, protocol
  `read`, replay `expect`, present — which is where device work is waited
  for ([residency test](crates/gpu/tests/parity.rs)).
- Presentation of a resident output is a blit on the device; without a
  surface the Player downloads and blits through softbuffer
  ([presenter](crates/gpu/src/present.rs)).
- `at_linear` is gated by layout (all floats, structs of floats mixed
  fieldwise), not by the word `Image`: bilinear on grids, trilinear on
  volumes, centres at half-integers, edges clamped.

### Execution modes

Five modes, each declared in source and carried in the manifest
([IR](crates/ir/src/lib.rs), [validator](crates/ir/src/validate.rs),
[declaration](crates/compiler/src/lower/declare.rs)); the determinism
rule of each is what the backends are held to
([parity](crates/gpu/tests/parity.rs), [matrix](crates/matrix/src/lib.rs)).

| Mode | Determinism rule | Not obvious |
| --- | --- | --- |
| map | per element | the only mode that may write several outputs: `compute` gives them by name as a record, the entry gives one value per output, and each lands in a Field of its own over the one domain ([lowering](crates/compiler/src/lower/mod.rs)); the GPU binds the further outputs after the inputs and the entry stores each itself ([WGSL](crates/gpu/src/wgsl.rs)) |
| reduce (`merge`) | a fixed pairwise tree, an odd last element carried up, [whatever the thread count](crates/rules/src/reduce.rs) | the GPU folds sixteen at a time, which reaches the same expressions as four levels; the CPU and JIT walk one level at a time on purpose. A `merge` may read Fields, params and time: the fold pass carries the map pass's bindings |
| scan (`scan with`) | element order, one thread / one invocation | a memory port is input and output at once; the bound Field is copied, never written, so a failed evaluation commits nothing. `init` may not `put`, checked over the call graph |
| scatter (`scatter into … with`) | as if the pairs were applied in source order; outside addresses drop | the GPU applies order-free rules with atomics (floats as one total order of keys, `last` as the greatest source index) and a float `sum` by a bitonic sort of pairs, the CPU's sequence exactly. A scatter may set its own `domain`; its output keeps the target's extent |
| prefix (`prefix with`) | any grouping, so only rules that give the same words in any: integer `sum`, `min`, `max` | a float `sum` is refused (that is `scan`), `last` too (that is a map). The GPU folds in three passes over blocks of about √n ([block](crates/rules/src/lib.rs)) — no shared memory, no barrier |

A reduce, scan, scatter or async node is never fused. A runtime may run
its scans on a second backend (`Runtime::with_backends`): a `gpu` session
hands scans to the JIT, bringing their inputs to the Host and uploading
their outputs when read ([test](crates/runtime/tests/mixed.rs)).

### Fusion, repetition, native operations

- Fusion inlines a producer into its only consumer when the consumer reads
  it only through `sample`, over the same domain, and the producer is not
  an output, a state source or pinned ([`fuse`](crates/ir/src/fuse.rs)).
  Bit-identical ([IR test](crates/ir/tests/fusion.rs)). Identical structs
  share one entry in the fused table, reused in order, or two arms of an
  `if` yield different struct indices and the module does not validate.
  A `Part` read of a fused input is written out as the extracts it
  stands for. A producer of several outputs fuses whole, each linked
  input taking the value of its output, when every reader of every
  output is the one consumer; otherwise it runs and writes them all
  ([`fuse_outputs`](crates/ir/src/fuse.rs), [planner](crates/graph/src/evaluate.rs),
  [composition tests](crates/scene/tests/composition.rs)).
- A backend answers with one Field per declared output or not at all:
  the runtime releases what a backend gave and fails the execution when
  the count differs ([`held_to`](crates/runtime/src/lib.rs),
  [test](crates/runtime/tests/hot_swap.rs)).
- Repetition is fusion without inlining: the same Kernel placed over its
  own output, same generation, params, bindings and domain, nothing else
  reading between, is one group ([planner](crates/graph/src/evaluate.rs));
  the GPU records the passes into one encoder with two buffers taking
  turns ([`Backend::execute_repeated`](crates/rules/src/backend.rs)).
- Native operations are device access published as versions at
  generation zero; a Kernel may not be published over one and none is
  fused ([`NativeOp`](crates/rules/src/backend.rs)). The rasterizer's
  contract ([rules](crates/rules/src/raster.rs)) has no clipping: a
  triangle with any `w <= 0` is skipped whole, so the device stage sends
  such a triangle to a point rather than clip it
  ([near-plane replay](examples/mesh/tests/near.replay)). The cutout form
  reads the mask by the shared `mask_alpha` formula so the edge is the
  same pixel on every backend.

### Graph, state and composition

- State lives in the Graph: `<~` reads what the previous successful
  evaluation committed; every node of one evaluation reads the same
  state; an evaluation commits atomically at its end or not at all; a
  republished Kernel inherits the state ([evaluator](crates/graph/src/evaluate.rs),
  [feedback tests](crates/scene/tests/composition.rs)).
- Migration: before nodes run, each committed state's shape and layout is
  compared with its port's (the comparison every evaluation already
  makes); on a mismatch the named `migrate` Kernel runs once, or the port
  restarts from its initial source, and a failed migration leaves the
  state as it was (`reconcile_state`, [agent record](examples/migrate/agent.jsonl)).
- `times = N` expands in scene lowering into `name/1..N` chained through
  `carry`; a graph param used as `N` is structural, so the session lowers
  the document again when it changes ([scene](crates/scene/src/lib.rs)).
  A filled Field (`node zeros: Rows<Float> { domain = 65; }`) is lowered
  to a one-parameter Kernel.
- `evaluate_for` runs only the named nodes, what they read, every `<~`
  source and never-produced outputs; replay and `advance` demand only
  what they watch.
- Declarations and functions cross files through `use`, resolved by the
  session through the scene lowering's `declare` callback; a changed
  library republishes the Kernels that inherit from it; one name declared
  twice is refused with both files named, and a `use` cycle is reported
  as a cycle ([session](crates/session/src/lib.rs),
  [session test](crates/cli/tests/session.rs)).
- Packages: a directory with `package.osn`; a scene's `use` expands to
  what the manifest uses once its versions fit, and files are
  canonicalized so a Kernel reached by two paths has one owner. The four
  version constants are printed by `orla version` and `status`.
- `size(port)` of another node's memory port is the memory's extent, not
  that node's domain.

### Domain

- A node's `domain` is a small language over its ports, params and the
  window ([graph](crates/graph/src/domain.rs)), accepted at scene lowering
  ([scene](crates/scene/src/domain.rs)), resolved when the node binds and
  at planning, and printed back so documents round-trip.
- `value(port)` is the one term that reads data: this frame's number from
  a producer that has already run. Reading it is a wait on a device, so
  the port is held to a row of scalars and `VALUE_MAX_ELEMENTS`, and
  several readers of one Field wait once
  ([`domain_values`](crates/graph/src/evaluate.rs)). Such a node's
  expression moves onto the fused node under fusion's port names
  (`renamed`). No backend knows about it.
- The window's size and scale are context and part of a node's cache
  key, so a resize re-executes exactly the nodes it sizes.

### The language

What the parser or compiler does alone, so nothing below it knows
([parser](crates/language/src/parser.rs), [lowering](crates/compiler/src/lower/mod.rs)):

- `if` without `else` guards the rest of the block; `match` ranges are
  their values written out (at most 256); `let { x, y }` binds once and
  reads fields; `break` makes the loop carry a value and a stopped flag;
  a swizzle is extracts and a construct; `&&` and `||` are an `if`, so
  the other side never runs (the optimizer writes cheap, pure ones back
  as one operation); generic and Field-parameter functions are
  instantiated per call signature; integer addresses are the float ones
  converted; the lowered builtins (`saturate`, `distance`, `mod`, `isnan`,
  …) are spellings of existing operations.

Represented in the IR, and so on every backend:

- Structs, fixed arrays (`subscript` clamps a run-time index), `Mat2/3/4`
  column-major, `IVec2/3`, structured `if`, bounded `loop` whose count is
  any `Int` read once and clamped to `MAX_LOOP_COUNT`, and `match` on an
  `Int` (the JIT uses a `br_table` over a dense range). Matrix formulas
  live once in [`orla_rules::matrix`](crates/rules/src/matrix.rs) over a
  trait, so CPU and JIT agree bit for bit.
- A table (an array known at compile time) is read-only data on the JIT
  and one `Words` instruction on the interpreter.
- Capabilities are compile-time diagnostics with positions; the validator
  is the backstop. Only nodes declaring `time.read` or `input.read`
  re-execute when time or input changes
  ([clock caching test](crates/scene/tests/composition.rs)).
- `orla help` is generated from the IR's typing rule and the lowering that
  dispatches builtins, so the listing is what compiles
  ([help](crates/compiler/src/help.rs)).
- Unknown names answer with the near miss ([`orla_abi::names`](crates/abi/src/names.rs)),
  and a name a Kernel cannot find is looked for beside the scene and in
  every known library, so the diagnostic names the `use` that would bring
  it in.

### Context and devices

- Input, pad, window and sound rate are context
  ([ABI](crates/abi/src/lib.rs), [`orla_abi::pad`](crates/abi/src/lib.rs)).
  A pad is sticks and buttons in the context, not a device a scene opens;
  the Player also presses keys from it, so older scenes play on one
  ([gamepad](crates/player/src/gamepad.rs)). Replay and the protocol
  drive every one of them.
- The sound card is a Player adapter ([audio](crates/player/src/audio.rs)):
  it claims the first output of one- or two-float rows, and its real-time
  callback only drains a lock-free ring — it never waits on the Graph or
  the GPU and never allocates. While a device plays, it paces evaluation.

### Files and tools

- Assets are inputs with files, re-read when their stamp changes
  ([file](crates/host/src/file.rs)): PNG, PPM, PGM, WAV, `.field`, `text`,
  `bytes` and `frames`. PNG, WAV and inflate are written with no
  dependency and checked against reference vectors
  ([png](crates/session/src/png.rs), [wav](crates/session/src/wav.rs),
  [inflate](crates/session/src/inflate.rs)). A format of one's own is a
  Kernel over `bytes`; formats other software writes keep their readers in
  `orla-import`, where dependencies are allowed.
- MIDI in and out are written from the format's definition; glTF comes in
  with transforms baked and goes out as GLB, bit for bit on the round
  trip; compressed audio comes in through symphonia and goes out only as
  WAV ([import](crates/import/src/), [export](crates/export/src/)).
- `orla render` steps a scene at a fixed rate into PNG frames and one WAV;
  video and lossy audio are ffmpeg's job.

### Interfaces

- The protocol is an interface over the session, not a new authority:
  `prepare` never publishes, `publish` checks an expected generation,
  `compose` rewrites the scene document so the file stays the source of
  truth ([protocol](crates/session/src/protocol.rs)). `--listen` feeds every
  connection to the one session; a connection that opens with an HTTP
  upgrade speaks RFC 6455, written with no dependency.
- The C ABI is the session in another process's memory: a Field is read
  in place as a pointer with extent, stride, shape and schema; strings
  live until the next call, a Field's words until the next evaluate,
  request or close; one thread per session; no panic crosses. The header
  is written by hand ([header](crates/ffi/include/orla.h)).
- Bridges are clients outside the workspace, in their program's language,
  speaking only the protocol and the Host's files.
- Application tests are data: `.replay` files, run on every backend by
  the verification in [HANDOFF](HANDOFF.md) ([replay](crates/cli/src/replay.rs)).

### Runtime and backends

- The three backends are peers: each implements the
  [`Backend`](crates/rules/src/backend.rs) contract in its own crate, and
  the Kernel Runtime depends on none of them and on no frontend. It takes
  validated IR; compiling is the caller's step before `prepare_module`,
  so a Kernel that does not compile never reaches the registry
  ([hot swap](crates/runtime/tests/hot_swap.rs)). Which backend runs is the
  session's choice ([session](crates/session/src/lib.rs)).
- `prepare` does the expensive work, so `publish` is a pointer swap
  ([`Backend`](crates/rules/src/backend.rs)). `prepare_later` moves it to one
  worker; `run`, `replay` and `check` prepare inline for determinism, and
  `serve` and the Player enable the worker after the first load. The old
  version runs until the ticket lands, and a group runs unfused until its
  fused version arrives, which changes no pixel.
- The interpreter is the reference: every value has a fixed word offset in
  the ABI's layout, the code is built once per prepared Kernel, rows run in
  parallel and are written in order, and the first failing element in
  row-major order is the one reported ([interpreter](crates/cpu/src/eval/compile.rs),
  [parallel rows test](crates/cpu/tests/execute.rs)).
- The JIT matches it bit for bit: same word layout, same integer checks
  recorded in a context struct, Field reads and transcendental functions
  through the same Rust helpers. It compiles maps and scans; reduce and
  scatter run its compiled entry under the CPU's own fold and combine
  rules, and native operations are the CPU's code.
- The GPU matches to rounding and does not report integer arithmetic
  failures. The JIT is the default backend; the Player takes the GPU, then
  the JIT, then the interpreter. `--backend cpu` and `trace` ask for the
  reference.
- Async nodes run on a Host of their own thread and are adopted when they
  land, keyed like a cache entry; the frame carries the last output
  meanwhile ([later](crates/runtime/src/later.rs)).
- A Kernel can be traced: `let` bindings are a side table
  (`Function.bindings`, renumbered through fusion) and `orla_cpu::trace`
  runs one element of the unoptimized Kernel and reads them back
  ([trace](crates/cpu/src/lib.rs)).
- The optimizer runs between lowering and validation with exact rewrites
  only — folding, sharing, dead values, inlining short functions,
  hoisting what cannot fail out of loops, running a literal loop at
  compile time, narrowing a `Part` read to the words it names — and is
  held to the unoptimized IR word for word over the feature matrix
  ([optimize](crates/ir/src/optimize.rs), [test](crates/cpu/tests/optimizer.rs)).
  Nothing that can fail moves or folds: a failure belongs to the element
  that has it.

### Measuring

- A profile's numbers are each node's own and say which kind they are:
  the device's timestamps where the driver honours them, the work plus one
  round trip where it resolves zeros (Metal through wgpu does), with
  `Backend::measurement_floor` subtracted from every line; a frame's cost
  is `bench`'s question, not the profile's
  ([runtime](crates/runtime/src/lib.rs), [bench](crates/session/src/bench.rs)).
- The GPU buffer pile serves the smallest buffer big enough, never more
  than twice the size; no emitted shader asks a buffer its length
  ([`Spare::take`](crates/gpu/src/lib.rs)).
- Handle liveness and node lookup are counts and indexes kept by the
  methods that touch them, with a `debug_assert` against the walk
  ([`cached_handles`](crates/graph/src/evaluate.rs), [`Graph::index`](crates/graph/src/lib.rs)).

## Device facts

Each is guarded by a test that fails without the workaround; do not
remove one without running it on the device.

- A scan step's first load of a memory element was hoisted out of the step
  loop by Metal: the loop has a `storageBarrier()` between steps
  ([parity](crates/gpu/tests/parity.rs)).
- One dispatch dimension allows 65,535 workgroups: long `Rows` are
  dispatched as a grid, the fold carried in the context
  ([wgsl](crates/gpu/src/wgsl.rs)).
- Metal divided a sixteen-bit slice of a word by a constant wrongly:
  every `/` and `%` multiplies in a one read from the context
  (`OPAQUE_ONE`) so the compiler cannot narrow it (matrix row `narrow division`).
- Native `u32 / u32` is wrong for dividends with the top bit set:
  `udiv` and `urem` go through the signed division on halved dividends.
- WGSL's float `%` rounds the quotient first: a Float remainder is an
  emitted `fmod` on a fused multiply-add.
- A driver's fast math may fold `inf * 0`: a NaN made by arithmetic on
  the GPU is not promised; one that arrives in a Field is seen alike.
- wgpu's error scopes are thread-local only with its `std` feature, which
  the GPU backend therefore enables.
- Parameter and context uniforms are rewritten per node with
  `queue.write_buffer`, which orders against submissions, not recording:
  batching a frame into one submission would make every node read the
  last node's uniforms.

## Measured and declined

Kept so they are not rediscovered; reopen with a new measurement.

- A node-level parallel scheduler: a map already uses every core, and a
  frame measured as the sum of its nodes on all three backends. `async`
  is what was missing instead.
- A `times` that stops on a small residual: a solver's step count is its
  author's budget, not the substrate's.
- `break` in hot loops: one carried word, about one percent on the ray
  march; the showcase keeps `until` there.
- Folding a reduction sixteen at a time on the CPU: a dependency chain,
  slower than the tree.
