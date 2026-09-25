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
| `.orla` compiles | [compiler tests](crates/frontend/compiler/tests/compile.rs) |
| Kernel calls Host only through ABI | [scope tests](crates/core/abi/src/lib.rs), [denied execution](crates/backends/cpu/tests/execute.rs), [validator capability checks](crates/core/ir/src/validate.rs) |
| Kernel hot-swaps without restart | [hot swap tests](crates/interface/session/tests/hot_swap.rs), [session tests](crates/interface/orla/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/interface/session/tests/composition.rs), [example scene](examples/grayscale/scene.osn) |

## Layers

The crates sit in four directories, by layer: `crates/core` (Host, ABI,
IR, rules, Kernel Runtime, Graph Runtime), `crates/frontend` (the `.orla`
language, its compiler, `.osn` scenes), `crates/backends` (interpreter,
JIT, GPU, and the conformance matrix they are held to) and
`crates/interface` (everything a person or a program drives). Which layer
may depend on which, and which crates may take a dependency from outside
the workspace, is checked by [a test](crates/interface/orla/tests/layers.rs).
Tests depend in the same directions, so a crate is tested without
building the layers above it: a test that needs a higher crate lives where
that crate is below it (a runtime or scene test that runs Kernels lives in
the session's tests). A backend's tests alone may also use the compiler,
the reference interpreter and the conformance matrix, which are what a
backend is held to.

| Layer | Implementation | Evidence |
| --- | --- | --- |
| Host Core | [host](crates/core/host/src/lib.rs) | handles, budgets, bounds; the `.field` file is a codec of [orla-formats](crates/interface/formats/src/file.rs), [its bytes pinned](crates/interface/formats/tests/field_file.rs) |
| Capability ABI | [abi](crates/core/abi/src/lib.rs), [names](crates/core/abi/src/names.rs) | grants checked before every Host call; [the capability set pinned to `ABI_VERSION`](crates/core/abi/tests/contract.rs) |
| Orla IR | [ir](crates/core/ir/src/lib.rs), [validation](crates/core/ir/src/validate.rs), [display](crates/core/ir/src/display.rs), [fusion](crates/core/ir/src/fuse.rs), [optimizer](crates/core/ir/src/optimize.rs) | typing, SSA order, capability coverage; [fusion bit-identity](crates/backends/cpu/tests/fusion.rs) |
| Kernel language | [language](crates/frontend/language/src/lib.rs), [parser](crates/frontend/language/src/parser.rs), [lexer](crates/frontend/language/src/lexer.rs) | syntax, precedence, positioned errors, near misses |
| Compiler | [compiler](crates/frontend/compiler/src/lib.rs), [lowering](crates/frontend/compiler/src/lower/), [help](crates/frontend/compiler/src/help.rs) | [compile tests](crates/frontend/compiler/tests/compile.rs), [the language surface at its version](crates/frontend/compiler/tests/surface.rs) |
| File writes | [writes](crates/interface/session/src/writes.rs): an output written `to` a file after the commit, into an allowed directory, whole; an input's `else` fallback substituted by the [evaluator](crates/core/graph/src/evaluate/mod.rs) | [session](crates/interface/session/tests/writes.rs), [lowering](crates/frontend/scene/src/lib.rs), [save replay](examples/save/tests/tally.replay) |
| Shared rules | [rules](crates/core/rules/src/lib.rs), [the `Backend` contract](crates/core/rules/src/backend.rs), [reduce tree](crates/core/rules/src/reduce.rs), [rasterizer contract and fast path](crates/core/rules/src/raster.rs) | every backend's parity test runs by them |
| Reference backend | [cpu](crates/backends/cpu/src/lib.rs), [word code](crates/backends/cpu/src/eval/) | [execution tests](crates/backends/cpu/tests/execute.rs), [optimizer tests](crates/backends/cpu/tests/optimizer.rs) |
| JIT backend | [jit](crates/backends/jit/src/lib.rs), [Cranelift emission](crates/backends/jit/src/codegen.rs) | [parity](crates/backends/jit/tests/parity.rs), [matrix](crates/backends/jit/tests/matrix.rs), every replay on `jit` |
| GPU backend | [gpu](crates/backends/gpu/src/lib.rs), [WGSL emission](crates/backends/gpu/src/wgsl/mod.rs), [rasterizer](crates/backends/gpu/src/raster.rs), [presenter](crates/backends/gpu/src/present.rs) | [parity](crates/backends/gpu/tests/parity.rs), [matrix](crates/backends/gpu/tests/matrix.rs); both skip without an adapter; [the names a shader declares](crates/backends/gpu/tests/names.rs), without one |
| Kernel Runtime | [runtime](crates/core/runtime/src/lib.rs), [off-frame runs](crates/core/runtime/src/later.rs); takes validated IR and the backend it is given | [hot swap](crates/interface/session/tests/hot_swap.rs), [async](crates/interface/session/tests/later.rs), [mixed clocks](crates/interface/session/tests/mixed.rs) |
| Scene documents | [scene](crates/frontend/scene/src/lib.rs), [domain expressions](crates/frontend/scene/src/domain.rs) | [lowering tests](crates/frontend/scene/src/lib.rs), [every form lowered as recorded](crates/frontend/scene/tests/surface.rs) |
| Graph Runtime | [graph](crates/core/graph/src/lib.rs), [evaluator](crates/core/graph/src/evaluate/mod.rs), [domain](crates/core/graph/src/domain.rs) | [composition tests](crates/interface/session/tests/composition.rs), written as documents |
| Session | [session](crates/interface/session/src/session.rs): [scene](crates/interface/session/src/scene.rs), [kernel ledger](crates/interface/session/src/kernels.rs), [inputs](crates/interface/session/src/inputs.rs), [source resolution](crates/interface/session/src/sources.rs), [observation](crates/interface/session/src/observe.rs); [bench](crates/interface/session/src/bench.rs), [file formats](crates/interface/formats/src/lib.rs) | [session tests](crates/interface/orla/tests/session.rs) |
| Creation protocol | [protocol](crates/interface/protocol/src/lib.rs): [inspect](crates/interface/protocol/src/inspect.rs), [create](crates/interface/protocol/src/create.rs), [run](crates/interface/protocol/src/run.rs), [fields](crates/interface/protocol/src/fields.rs), [wire forms](crates/interface/protocol/src/wire.rs), [transport](crates/interface/protocol/src/transport.rs), [WebSocket](crates/interface/protocol/src/websocket.rs), [contract](docs/PROTOCOL.md) | [client test](crates/interface/orla/tests/protocol.rs); [live agents](crates/interface/orla/tests/agent.rs) repair, publish and compose from the answers alone, and change what a running state carries without losing it; the op table held to the server ([contract test](crates/interface/protocol/src/lib.rs)); [an agent's record](examples/scanline/agent.jsonl) replayed |
| Command line | [orla](crates/interface/orla/src/main.rs), [replay](crates/interface/session/src/replay.rs), [render](crates/interface/orla/src/render.rs), [packages](crates/interface/orla/src/package.rs), [scaffold](crates/interface/orla/src/scaffold.rs) | every example replay on three backends |
| C ABI | [ffi](crates/interface/ffi/src/lib.rs), [header](crates/interface/ffi/include/orla.h) | [ABI tests](crates/interface/ffi/tests/abi.rs), [header held to the exports](crates/interface/ffi/tests/header.rs), [C example](bridges/c/hello.c) |
| Player | [the play](crates/interface/session/src/play/mod.rs) every interface shares (frame, keys, preview, pads, F10 line, replay); the window's [loop](crates/interface/player/src/main.rs), [audio](crates/interface/player/src/audio.rs), [gamepad](crates/interface/player/src/gamepad.rs); a page's, [player.js](bridges/web/player.js) over the C ABI's [`orla_play_*`](crates/interface/ffi/src/play.rs) | [play tests](crates/interface/session/tests/play.rs), [ABI test](crates/interface/ffi/tests/abi.rs), the web bridge's test, presentation and blit tests |
| Import and export | [import](crates/interface/import/src/), [export](crates/interface/export/src/) | [MIDI tests](crates/interface/formats/tests/midi.rs), [round trip](crates/interface/export/tests/round_trip.rs) |
| Bridges | [bridges](bridges/), each in its program's own language | [bridge tests](crates/interface/orla/tests/bridges.rs), run when the program is present |
| Conformance matrix | [conformance matrix](crates/backends/conformance/src/lib.rs) | [on the JIT](crates/backends/jit/tests/matrix.rs) bit for bit, [on the GPU](crates/backends/gpu/tests/matrix.rs) ints exact and floats to rounding; the 64-bit rows on the JIT, the `Long` rows on both; the element shapes a scatter's target may hold (one word to twenty-one, `Long`s, rows and a volume, every rule, emitting where it may) from one narrow input, on both |

## Libraries, not primitives

What these packages do needed no engine feature; keep it that way.

| Library | Built from | Evidence |
| --- | --- | --- |
| [Stream compaction](packages/compact.orlapkg/kernels/compact.orla): a rank is `prefix with sum`, the count a reduce, the placement a scatter whose outside addresses drop | three execution modes and `value(port)` | [culled replay](examples/culled/tests/culled.replay), the same picture culled and not |
| [Radix sort](packages/sort.orlapkg/kernels/sort.orla) | prefix and scatter | [sort replay](examples/sort/tests/sort.replay) |
| [Mip levels](packages/picture.orlapkg/graphs/mip.osn): a sub-graph placed `levels` times with `carry`, the level read off a rasterized surface's neighbouring `uv` | `times`, `carry`, the rasterizer's `uv` and `id` | [mipmap replay](examples/mipmap/tests/ground.replay) |
| [Text](packages/text.orlapkg/kernels/text.orla): labels are `Rows<Glyph>`, the font a `Grid<Float>` atlas baked by a Kernel | shared functions, assets | [text replay](examples/text/tests/text.replay) |
| [Skinning](packages/skin.orlapkg/kernels/skin.orla): a rig's rows under the [rig contract](docs/RIGS.md); a joint a Kernel turns onto a point (`skin.aim@1`) over what the clip says, in its parent's space worked out up the chain; root motion, a travelling clip played in place (`skin.stride@1`) and its travel (`skin.travel@1`) carried by the scene through `<~` | maps over rows, feedback | [rig tests](crates/interface/orla/tests/rigs.rs), [skin](examples/skin/tests/skin.replay), [crowd](examples/crowd/tests/crowd.replay) and [reach](examples/reach/tests/reach.replay) replays; [a joint off the root](examples/reach/tests/elbow.replay) checked from the palette alone; [stride](examples/stride/tests/stride.replay), where the body stands the same at five frames or fifty |
| [Rasterization](packages/raster.orlapkg/kernels/raster.orla): each triangle walks its box of pixel centres and emits a `Long` key per pixel it covers, the depth's `bits` above and its row below; a scatter `min` keeps the nearest and, at one depth, the earliest | scatter with `emit`, 64-bit atomics on the GPU | [edges replay](examples/mesh/tests/edges.replay) on three backends; [the fast path held to the rule](crates/core/rules/src/raster.rs); a cutout (`raster.cover_cutout@2`), held to `native.rasterize@2` by [a replay](examples/mesh/tests/cutout.replay) on three backends and to the bit off the device by [a session test](crates/interface/orla/tests/session.rs); tiled presets, which cover a large triangle a tile to an invocation (a prefix of tile counts, and a pool that reads the total on the device), held to the one-pass forms to the bit by the same replay; [ten million triangles at 4K](examples/mesh/tests/ten_million.osn), for `bench` |

## Contracts

### Resource and residency

- One resource, following [the Typed Field decision](docs/TYPED-FIELD-ARCHITECTURE.md):
  a Field of `Rows`, `Grid` or `Volume`, one 32-bit word per scalar
  ([host](crates/core/host/src/lib.rs)). Port types are structural; `Image` is
  the compiler's alias for `Grid<Vec4>`. Bindings are checked by shape and
  layout, never by spelling ([evaluator](crates/core/graph/src/evaluate/bind.rs)).
  Every backend carries elements as words, so integers and booleans are
  exact everywhere.
- A `Double` or a `Long` is two words, the low one first, everywhere a
  word is the unit: the schema lists both (`d` and `l` in its text, and
  a word without its partner is refused), a Value keeps an `f64` or an
  `i64`, and no narrow encoding fits one ([schema](crates/core/host/src/lib.rs)).
  The interpreter has instructions of its own for them
  ([word code](crates/backends/cpu/src/eval/code.rs)); the JIT holds each as one
  `F64` or `I64` value and lays memory out by bytes
  ([emission](crates/backends/jit/src/codegen.rs)). A number written without a
  width is exact until it meets one: the compiler adapts it where it
  meets a 64-bit operand, argument, field or branch
  ([`adapt`](crates/frontend/compiler/src/lower/mod.rs)); a value given from
  outside converts by [`Number::as_kind`](crates/core/abi/src/lib.rs), a
  float given to a double taken as the decimal it reads as. Held by
  [values against Rust's own](crates/backends/cpu/tests/wide.rs), the matrix's
  64-bit rows, the [protocol round trip](crates/interface/protocol/src/lib.rs)
  and the [precision example](examples/precision/README.md).
- A scalar may be kept narrower than a word (`Grid<Vec4 as unorm8>`):
  the schema carries each scalar's [`Encoding`](crates/core/host/src/encoding.rs)
  and byte offset, elements pack with nothing between, and every scalar
  sits at a multiple of its size, so none crosses a word
  ([schema](crates/core/host/src/lib.rs)). The widening and narrowing rule is
  written once, in the Host, and held three ways: the interpreter calls
  it, the JIT emits the same loads and calls it for halves, the GPU
  emits the same integer arithmetic ([encoding tests](crates/core/host/src/encoding.rs)
  over every half, [conformance matrix](crates/backends/conformance/src/lib.rs) `narrow` rows on
  both backends, the GPU's stored bits compared exactly). A Kernel's
  words and a Field's storage are different things in the API:
  `values()` and `write_values` are the former, `storage()` and
  `write_storage` the latter. Budgets, downloads and `.field` files
  count storage; the protocol's words and the C ABI's are values, the
  C ABI handing a narrow Field over widened in the session's memory.
  Only a map or a scan writes a narrow output, and a narrow link is
  never fused ([validator](crates/core/ir/src/validate.rs),
  [planner](crates/core/graph/src/evaluate/plan.rs),
  [composition test](crates/interface/session/tests/composition.rs), the
  [packed example](examples/packed/README.md) and its
  [session test](crates/interface/orla/tests/session.rs)).
- The Host owns every Field; a slot may hold its words on a device as an
  object only the backend understands. The GPU binds resident inputs,
  uploads Host inputs once, and leaves outputs resident, so committed state
  ping-pongs on the device. Words return only at observation boundaries
  through [`Runtime::image`](crates/core/runtime/src/lib.rs) — export, protocol
  `read`, replay `expect`, present — which is where device work is waited
  for ([residency test](crates/backends/gpu/tests/parity.rs)).
- Presentation of a resident output is a blit on the device; without a
  surface the Player downloads and blits through softbuffer
  ([presenter](crates/backends/gpu/src/present.rs)).
- `at_linear` is gated by layout (all floats, structs of floats mixed
  fieldwise), not by the word `Image`: bilinear on grids, trilinear on
  volumes, centres at half-integers, edges clamped.

### Execution modes

Five modes, each declared in source and carried in the manifest
([IR](crates/core/ir/src/lib.rs), [validator](crates/core/ir/src/validate.rs),
[declaration](crates/frontend/compiler/src/lower/declare.rs)); the determinism
rule of each is what the backends are held to
([parity](crates/backends/gpu/tests/parity.rs), [conformance matrix](crates/backends/conformance/src/lib.rs)).

| Mode | Determinism rule | Not obvious |
| --- | --- | --- |
| map | per element | the only mode that may write several outputs: `compute` gives them by name as a record, the entry gives one value per output, and each lands in a Field of its own over the one domain ([lowering](crates/frontend/compiler/src/lower/mod.rs)); the GPU binds the further outputs after the inputs and the entry stores each itself ([WGSL](crates/backends/gpu/src/wgsl/map.rs)) |
| reduce (`merge`) | a fixed pairwise tree, an odd last element carried up, [whatever the thread count](crates/core/rules/src/reduce.rs) | the GPU folds sixteen at a time, which reaches the same expressions as four levels; the CPU and JIT walk one level at a time on purpose, the JIT merging a level's pairs across workers ([parity](crates/backends/jit/tests/parity.rs)). A `merge` may read Fields, params and time: the fold pass carries the map pass's bindings |
| scan (`scan with`) | element order, one thread / one invocation | a memory port is input and output at once; the bound Field is copied, never written, so a failed evaluation commits nothing. `init` may not `put`, checked over the call graph |
| scatter (`scatter into … with`) | as if the pairs were applied in source order; outside addresses drop | the GPU applies order-free rules with atomics (floats as one total order of keys, `last` as the greatest source index) and a float `sum` by a bitonic sort of pairs, the CPU's sequence exactly. A scatter may set its own `domain`; its output keeps the target's extent. `emit` gives more pairs under an order-free rule only, and the GPU lands them, and the pair the entry returns, with atomics from the entry itself, the target written from its base before the entry runs (float and `Long` words turned to keys as they are written) ([WGSL](crates/backends/gpu/src/wgsl/scatter.rs); [a keyed target from its base](crates/backends/gpu/tests/parity.rs)) |
| prefix (`prefix with`) | any grouping, so only rules that give the same words in any: integer `sum`, `min`, `max` | a float `sum` is refused (that is `scan`), `last` too (that is a map). The GPU folds in three passes over blocks of about √n ([block](crates/core/rules/src/lib.rs)) — no shared memory, no barrier; the JIT folds runs across workers the same way, and hands a run whose total fails to the walk, which names the failure ([JIT](crates/backends/jit/src/lib.rs)) |

A reduce, scan, scatter or async node is never fused. A runtime may run
its scans on a second backend (`Runtime::with_backends`): a `gpu` session
hands scans to the JIT, and every module the GPU does not take
([`Backend::takes`](crates/core/rules/src/backend.rs)): one binding more buffers
at once than the device does even with its read-only inputs packed (a
Kernel wider than the device's bindings reads the inputs past them from
one buffer, each at its offset, copied there before its passes run —
[packing](crates/backends/gpu/src/wgsl/mod.rs); what cannot be packed is a
memory, which its steps write), one whose branches and loops nest past
the 127 braces a shader may (known from the IR before a shader is
emitted), one holding a `Double`,
or a `Long` outside a map, a reduction, or a scatter of `Long`s under
`min` or `max` (64-bit atomics on keys whose unsigned order is the signed
one), or on a device without 64-bit integers. Their inputs come to the Host and their outputs upload when
read ([test](crates/interface/session/tests/mixed.rs)). A report says where every
node ran, why when it left the main backend, and the bytes the trip
carried ([`Runtime::placement`](crates/core/runtime/src/lib.rs),
[traffic test](crates/backends/gpu/tests/parity.rs)); a session says it when the
scene loads, once per change ([routed test](crates/interface/orla/tests/session.rs)),
and a replay can pin it (`expect placed`). On the device a `Long` is
WGSL's `i64`; naga's own 64-bit `/` and `%` go through `select`, which
Metal cannot pick for a `long`, so the preamble writes its own division
and nothing emitted for a `Long` may use `select`
([WGSL](crates/backends/gpu/src/wgsl/prelude.rs)).

### Fusion, repetition, native operations

- Fusion inlines a producer into its only consumer when the consumer reads
  it only through `sample`, over the same domain, and the producer is not
  an output, a state source or pinned ([`fuse`](crates/core/ir/src/fuse.rs)).
  Bit-identical ([IR test](crates/backends/cpu/tests/fusion.rs)). Identical structs
  share one entry in the fused table, reused in order, or two arms of an
  `if` yield different struct indices and the module does not validate.
  A `Part` read of a fused input is written out as the extracts it
  stands for. A producer of several outputs fuses whole, each linked
  input taking the value of its output, when every reader of every
  output is the one consumer; otherwise it runs and writes them all
  ([`fuse_outputs`](crates/core/ir/src/fuse.rs), [planner](crates/core/graph/src/evaluate/plan.rs),
  [composition tests](crates/interface/session/tests/composition.rs)).
- A backend answers with one Field per declared output or not at all:
  the runtime releases what a backend gave and fails the execution when
  the count differs ([`held_to`](crates/core/runtime/src/lib.rs),
  [test](crates/interface/session/tests/hot_swap.rs)).
- Repetition is fusion without inlining: the same Kernel placed over its
  own output, same generation, params, bindings and domain, nothing else
  reading between, is one group ([planner](crates/core/graph/src/evaluate/plan.rs));
  the GPU records the passes into one encoder with two buffers taking
  turns ([`Backend::execute_repeated`](crates/core/rules/src/backend.rs)).
- Native operations are device access published as versions at
  generation zero; a Kernel may not be published over one and none is
  fused ([`NativeOp`](crates/core/rules/src/backend.rs)). The rasterizer's
  contract ([rules](crates/core/rules/src/raster.rs)) has no clipping: a
  triangle with any `w <= 0` is skipped whole, so the device stage sends
  such a triangle to a point rather than clip it
  ([near-plane replay](examples/mesh/tests/near.replay)). The cutout form
  reads the mask by the shared `mask_alpha` formula so the edge is the
  same pixel on every backend. A native operation names the structs of
  its ports ([`NativeOp::structs`](crates/core/rules/src/backend.rs)),
  so its elements are read by field as a Kernel's are
  ([native replay](examples/mesh/tests/native.replay)).
- The raster package is the rasterizer a scene places; `native.rasterize@1`
  is its fast path ([decision](docs/DECISIONS.md)). On the interpreter
  and the JIT the fast path sets triangles up in parallel runs and draws
  bands of rows in parallel, and a test holds it bit for bit to the rule
  written plainly; on the GPU it is the device's rasterizer.

### Graph, state and composition

- State lives in the Graph: `<~` reads what the previous successful
  evaluation committed; every node of one evaluation reads the same
  state; an evaluation commits atomically at its end or not at all; a
  republished Kernel inherits the state ([evaluator](crates/core/graph/src/evaluate/state.rs),
  [feedback tests](crates/interface/session/tests/composition.rs)).
- Migration: before nodes run, each committed state's shape and layout is
  compared with its port's (the comparison every evaluation already
  makes); on a mismatch the named `migrate` Kernel runs once, or the port
  restarts from its initial source, and a failed migration leaves the
  state as it was (`reconcile_state`, [agent record](examples/migrate/agent.jsonl)).
- `times = N` expands in scene lowering into `name/1..N` chained through
  `carry`; a graph param used as `N` is structural, so the session lowers
  the document again when it changes ([scene](crates/frontend/scene/src/lib.rs)).
  A filled Field (`node zeros: Rows<Float> { domain = 65; }`) is lowered
  to a one-parameter Kernel.
- `evaluate_for` runs only the named nodes, what they read, every `<~`
  source and never-produced outputs; replay and `advance` demand only
  what they watch.
- Declarations and functions cross files through `use`, resolved by the
  session through the scene lowering's `declare` callback; a changed
  library republishes the Kernels that inherit from it; one name declared
  twice is refused with both files named, and a `use` cycle is reported
  as a cycle ([source resolution](crates/interface/session/src/sources.rs),
  [session test](crates/interface/orla/tests/session.rs)).
- Packages: a directory with `package.osn`; a scene's `use` expands to
  what the manifest uses once its versions fit, and files are
  canonicalized so a Kernel reached by two paths has one owner. The four
  version constants are printed by `orla version` and `status`.
- A parameter is any value type, carried as its words: the manifest
  gives each [`Param`](crates/core/ir/src/lib.rs) its type, a
  [`Value`](crates/core/abi/src/lib.rs) is a matrix, an integer vector or a
  `Record` of a struct's fields or an array's elements, and a value is
  checked against a type by layout alone ([`value_as`](crates/core/ir/src/lib.rs),
  `Value::shaped_like`): an integer stands in for a float, and a record
  given by position takes the parameter's shape. That one rule serves
  a Kernel's default ([compiler](crates/frontend/compiler/src/lower/declare.rs)
  `literal_in`), a scene's literal, a graph parameter set by a replay
  or the protocol ([scene](crates/interface/session/src/scene.rs) `set_param`),
  and the binding of a node ([evaluator](crates/core/graph/src/evaluate/bind.rs)).
  The interpreter and the JIT read a parameter's words where the params
  are laid end to end; the GPU keeps them as `vec4<u32>` slots, each
  parameter from a slot of its own, and builds the value from its words
  ([wgsl](crates/backends/gpu/src/wgsl/storage.rs) `param_slots`), held by the
  [conformance matrix](crates/backends/conformance/src/lib.rs) row of such parameters under every
  mode and the [cube](examples/cube/README.md) replay.
- `size(port)` of another node's memory port is the memory's extent, not
  that node's domain.

### Domain

- A node's `domain` is a small language over its ports, params and the
  window ([graph](crates/core/graph/src/domain.rs)), accepted at scene lowering
  ([scene](crates/frontend/scene/src/domain.rs)), resolved when the node binds and
  at planning, and printed back so documents round-trip.
- `value(port)` is the one term that reads data: this frame's number from
  a producer that has already run. Reading it is a wait on a device, so
  the port is held to a row of scalars and `VALUE_MAX_ELEMENTS`, and
  several readers of one Field wait once
  ([`domain_values`](crates/core/graph/src/evaluate/bind.rs)). Such a node's
  expression moves onto the fused node under fusion's port names
  (`renamed`). No backend knows about it.
- The window's size and scale are context and part of a node's cache
  key, so a resize re-executes exactly the nodes it sizes.

### The language

What the parser or compiler does alone, so nothing below it knows
([parser](crates/frontend/language/src/parser.rs), [lowering](crates/frontend/compiler/src/lower/mod.rs)):

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
  `Int` (the JIT uses a `br_table` over a dense range).
- A Kernel's work is bounded twice: [`Manifest::budget`](crates/core/ir/src/lib.rs)
  is the loop iterations one element may run, and
  [`Host::work`](crates/core/host/src/lib.rs) the iterations one run may take,
  all elements together, kept as a [`Run`](crates/core/rules/src/lib.rs) that
  elements hand whole blocks to. [`Module::counts_work`](crates/core/ir/src/lib.rs)
  counts only when the static bound (`worst_work`) can fill a block or
  pass the budget. Where work is counted is IR:
  [`account`](crates/core/ir/src/account.rs) places `Op::Spend` after every
  loop and `Op::Look` where an element looks, and the runtime accounts a
  module as it prepares it ([`prepare_on`](crates/core/runtime/src/lib.rs)),
  so a backend implements two operations and never the rule. It fails
  with `ExecutionError::Exhausted` at the same element, or `Overrun` for
  the run, whose sum is exact
  ([IR tests](crates/frontend/compiler/tests/compile.rs),
  [session tests](crates/interface/orla/tests/session.rs)).
- Integer failures are the reference's on every backend. The GPU emits
  checked arithmetic for a module that can fail ([`can_fail`](crates/backends/gpu/src/wgsl/mod.rs)),
  each invocation recording its first failure into one word by
  `atomicMax`, so the least element wins; `execute_checked` hands back a
  [`Check`](crates/core/rules/src/backend.rs) and the evaluator waits on every
  check before it commits, dropping what the failed node and the nodes
  after it cached ([device test](crates/backends/gpu/tests/parity.rs),
  [session test](crates/interface/orla/tests/session.rs)). An arm that can fail is
  never computed speculatively.
- Types are structural in the ABI: a scalar of a kind, a vector, a
  matrix or a struct, with a constant for every name a source writes
  ([`Type`](crates/core/abi/src/lib.rs)).
- The builtins that are formulas over the arithmetic are not operations
  of the IR: products of matrices, `dot`, `cross`, `length`,
  `normalize`, `reflect`, `transpose`, `determinant` and `inverse` are
  written into it by the compiler as the arithmetic they are
  ([formulas](crates/frontend/compiler/src/lower/composite.rs),
  [determinant and inverse](crates/frontend/compiler/src/lower/matrix.rs)), so a
  backend implements none of them and all three compute one formula in
  one order. The GPU chooses between two short arms of plain arithmetic
  with `select` rather than a branch ([WGSL](crates/backends/gpu/src/wgsl/emitter.rs)),
  which is what keeps `normalize` as fast as the device's own.
- A table (an array known at compile time) is read-only data on the JIT
  and one `Words` instruction on the interpreter.
- Capabilities are compile-time diagnostics with positions; the validator
  is the backstop. Only nodes declaring `time.read` or `input.read`
  re-execute when time or input changes
  ([clock caching test](crates/interface/session/tests/composition.rs)).
- `orla help` is generated from the IR's typing rule and the lowering that
  dispatches builtins, so the listing is what compiles
  ([help](crates/frontend/compiler/src/help.rs)).
- Unknown names answer with the near miss ([`orla_abi::names`](crates/core/abi/src/names.rs)),
  and a name a Kernel cannot find is looked for beside the scene and in
  every known library, so the diagnostic names the `use` that would bring
  it in.

### Context and devices

- Input, pad, window and sound rate are context
  ([ABI](crates/core/abi/src/lib.rs), [`orla_abi::pad`](crates/core/abi/src/lib.rs)).
  A pad is sticks and buttons in the context, not a device a scene opens;
  the play also presses keys from it, so older scenes play on one
  ([pad](crates/interface/session/src/play/pad.rs)); each device names its
  pads' buttons and axes to it. Replay and the protocol
  drive every one of them.
- The sound card is a Player adapter ([audio](crates/interface/player/src/audio.rs)):
  the play names the first output of one- or two-float rows, and its real-time
  callback only drains a lock-free ring — it never waits on the Graph or
  the GPU and never allocates. While a device plays, it paces evaluation.

### Files and tools

- Assets are inputs with files, re-read when their stamp changes
  ([formats](crates/interface/formats/src/lib.rs)): PNG, PPM, PGM, WAV, MIDI, `.field`, `text`,
  `bytes` and `frames`. PNG, WAV and inflate are written with no
  dependency and checked against reference vectors
  ([png](crates/interface/formats/src/png.rs), [wav](crates/interface/formats/src/wav.rs),
  [inflate](crates/interface/formats/src/inflate.rs)). Which formats are read live
  and which are converted is [a decision](docs/DECISIONS.md) and
  [`orla-formats`](crates/interface/formats/src/lib.rs) holds the ones read live.
- MIDI in and out are written from the format's definition
  ([midi](crates/interface/formats/src/midi.rs)), and an input may name a `.mid`
  itself; glTF comes in
  with transforms baked and goes out as GLB, bit for bit on the round
  trip; compressed audio comes in through symphonia and goes out only as
  WAV ([import](crates/interface/import/src/), [export](crates/interface/export/src/)).
- `orla render` steps a scene at a fixed rate into PNG frames and one WAV;
  video and lossy audio are ffmpeg's job.

### Interfaces

- The protocol is an interface over the session, not a new authority:
  `prepare` never publishes, `publish` checks an expected generation,
  `compose` rewrites the scene document so the file stays the source of
  truth ([protocol](crates/interface/protocol/src/lib.rs)). `--listen` feeds every
  connection to the one session; a connection that opens with an HTTP
  upgrade speaks RFC 6455, written with no dependency.
- The C ABI is the session in another process's memory: a Field is read
  in place as a pointer with extent, stride, shape and schema; strings
  live until the next call, a Field's words until the next evaluate,
  request or close; one thread per session; no panic crosses. The header
  is written by hand ([header](crates/interface/ffi/include/orla.h)).
- Bridges are clients outside the workspace, in their program's language,
  speaking only the protocol and the Host's files.
- Application tests are data: `.replay` files, run on every backend by
  the verification in [HANDOFF](HANDOFF.md) ([replay](crates/interface/session/src/replay.rs)).

### Runtime and backends

- The three backends are peers: each implements the
  [`Backend`](crates/core/rules/src/backend.rs) contract in its own crate, and
  the Kernel Runtime depends on none of them and on no frontend. It takes
  validated IR; compiling is the caller's step before `prepare_module`,
  so a Kernel that does not compile never reaches the registry
  ([hot swap](crates/interface/session/tests/hot_swap.rs)). Which backend runs is the
  session's choice ([session](crates/interface/session/src/session.rs) `runtime_for`).
- `prepare` does the expensive work, so `publish` is a pointer swap
  ([`Backend`](crates/core/rules/src/backend.rs)). `prepare_later` moves it to one
  worker; `run`, `replay` and `check` prepare inline for determinism, and
  `serve` and the Player enable the worker after the first load. The old
  version runs until the ticket lands, and a group runs unfused until its
  fused version arrives, which changes no pixel.
- The interpreter is the reference: every value has a fixed word offset in
  the ABI's layout, the code is built once per prepared Kernel, rows run in
  parallel and are written in order, and the first failing element in
  row-major order is the one reported ([interpreter](crates/backends/cpu/src/eval/compile.rs),
  [parallel rows test](crates/backends/cpu/tests/execute.rs)).
- The JIT matches it bit for bit: same word layout, same integer checks
  recorded in a context struct, Field reads and transcendental functions
  through the same Rust helpers. It compiles maps and scans; reduce and
  scatter run its compiled entry under the fold and combine rules of
  [rules](crates/core/rules/src/lib.rs), and native operations are the
  reference rasterizer there; the JIT takes nothing from the interpreter.
- The GPU matches to rounding. The JIT is the default backend; the Player takes the GPU, then
  the JIT, then the interpreter. `--backend cpu` and `trace` ask for the
  reference.
- Async nodes run on a Host of their own thread and are adopted when they
  land, keyed like a cache entry; the frame carries the last output
  meanwhile ([later](crates/core/runtime/src/later.rs)).
- A Kernel can be traced: `let` bindings are a side table
  (`Function.bindings`, renumbered through fusion) and `orla_cpu::trace`
  runs one element of the unoptimized Kernel and reads them back
  ([trace](crates/backends/cpu/src/lib.rs)).
- The optimizer runs between lowering and validation with exact rewrites
  only — folding, sharing, dead values, inlining short functions,
  hoisting what cannot fail out of loops, running a literal loop at
  compile time, narrowing a `Part` read to the words it names — and is
  held to the unoptimized IR word for word over the feature matrix
  ([optimize](crates/core/ir/src/optimize.rs), [test](crates/backends/cpu/tests/optimizer.rs)).
  Nothing that can fail moves or folds: a failure belongs to the element
  that has it.

- A platform without a primitive does the same work another way: where
  no thread can be started (WebAssembly), an async run is done before
  `start_run` returns ([later](crates/core/runtime/src/later.rs)), the
  preparation worker is not started, and `serve` reads each request in
  turn; where links cannot be resolved, a file's one name is its path
  folded ([store](crates/interface/session/src/store.rs)); where
  nothing loads the code a JIT writes, the session has no JIT; where
  there is no clock (a page), nothing is timed
  ([clock](crates/core/runtime/src/clock.rs)). The replays run under
  `wasmtime` on the interpreter (HANDOFF's verification).
- Where Orla is WebAssembly, the JIT writes WebAssembly
  ([wasm](crates/backends/jit/src/wasm)) that the host loads into Orla's
  function table, and everything around the code generator is the
  native JIT's. Held by the JIT's own tests and the core's, the front
  end's and the interpreter's, run as 32-bit WebAssembly in node
  ([wasm test](crates/interface/orla/tests/wasm.rs)).
- In a page the GPU backend is the browser's WebGPU through wgpu, and it
  waits for the device by parking ([wait](crates/backends/gpu/src/wait.rs)):
  the page suspends Orla until the device's answer wakes it. A picture it holds is drawn on a page's canvas with
  the Player's own presenter (`orla_present`), nothing read back. Held
  by the web bridge's test in a headless Chrome.
- A session reads and writes its documents through a
  [`Store`](crates/interface/session/src/store.rs): the disk, or the
  files its caller handed in (`Memory`), which is how Orla runs in a
  page with no disk; the C ABI opens a session on them (`orla_open_files`)
  and the web bridge's test runs the text example in a headless Chrome
  that way, composing it and handing a changed document in
  ([bridges test](crates/interface/orla/tests/bridges.rs),
  [ABI test](crates/interface/ffi/tests/abi.rs)).

### Measuring

- A profile's numbers are each node's own and say which kind they are:
  the device's timestamps where the driver honours them, the work plus one
  round trip where it resolves zeros (Metal through wgpu does), with
  `Backend::measurement_floor` subtracted from every line; a frame's cost
  is `bench`'s question, not the profile's
  ([runtime](crates/core/runtime/src/lib.rs), [bench](crates/interface/session/src/bench.rs)).
- The GPU buffer pile serves the smallest buffer big enough, never more
  than twice the size; no emitted shader asks a buffer its length
  ([`Spare::take`](crates/backends/gpu/src/lib.rs)).
- Handle liveness and node lookup are counts and indexes kept by the
  methods that touch them, with a `debug_assert` against the walk
  ([`cached_handles`](crates/core/graph/src/evaluate/ledger.rs), [`Graph::index`](crates/core/graph/src/lib.rs)).

## Device facts

Each is guarded by a test that fails without the workaround; do not
remove one without running it on the device.

- A scan step's first load of a memory element was hoisted out of the step
  loop by Metal: the loop has a `storageBarrier()` between steps
  ([parity](crates/backends/gpu/tests/parity.rs)).
- One dispatch dimension allows 65,535 workgroups: long `Rows` are
  dispatched as a grid, the fold carried in the context
  ([wgsl](crates/backends/gpu/src/wgsl/mod.rs)).
- Metal divided a sixteen-bit slice of a word by a constant wrongly:
  every `/` and `%` multiplies in a one read from the context
  (`OPAQUE_ONE`) so the compiler cannot narrow it (matrix row `narrow division`).
- Native `u32 / u32` is wrong for dividends with the top bit set:
  `udiv` and `urem` go through the signed division on halved dividends.
- WGSL's float `%` rounds the quotient first: a Float remainder is an
  emitted `fmod` on a fused multiply-add.
- A driver's fast math may fold `inf * 0`: a NaN made by arithmetic on
  the GPU is not promised; one that arrives in a Field is seen alike.
- A storage buffer is written a 32-bit word at a time, so neighbouring
  elements of a narrow output that share a word would race: those
  outputs are declared `atomic<u32>`, cleared, and stored by `atomicOr`;
  outputs whose elements fill whole words are stored a word at a time
  ([wgsl](crates/backends/gpu/src/wgsl/storage.rs) `narrow_store_body`). The encodings are
  integer arithmetic and bit-exact constants rather than `pack2x16float`
  and division, which a driver's fast math may round its own way.
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
