# Implementation evidence

**Answers:** where the code is, and which test establishes each contract.

Every line points at code or a test; the test is the claim, this is the
index to it. Syntax is in [DOCUMENTS](docs/DOCUMENTS.md), the wire in
[PROTOCOL](docs/PROTOCOL.md), commands in [USAGE](docs/USAGE.md), reasons
in [DECISIONS](docs/DECISIONS.md).

## The four proofs

| Proof | Where it is exercised |
| --- | --- |
| `.ent` compiles | [compiler tests](crates/frontend/compiler/tests/compile.rs) |
| Kernel calls Host only through ABI | [scope tests](crates/core/abi/src/lib.rs), [denied execution](crates/backends/cpu/tests/execute.rs), [validator capability checks](crates/core/ir/src/validate.rs) |
| Kernel hot-swaps without restart | [hot swap tests](crates/interface/session/tests/hot_swap.rs), [a scene's change swapped whole](crates/interface/session/tests/scene_change.rs), [session tests](crates/interface/entelechy/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/interface/session/tests/composition.rs), [example scene](examples/grayscale/scene.esn) |

## Layers

The crates sit by layer in `crates/core`, `crates/frontend`,
`crates/backends` and `crates/interface`. Which layer depends on which,
and which crates take an outside dependency, is held by
[a test](crates/interface/entelechy/tests/layers.rs). Tests depend in the same
directions: a test needing a higher crate lives in that crate's tests.

| Layer | Implementation | Evidence |
| --- | --- | --- |
| Host Core | [host](crates/core/host/src/lib.rs) | handles, budgets, bounds; `.field` in [entelechy-formats](crates/interface/formats/src/file.rs), [its bytes pinned](crates/interface/formats/tests/field_file.rs) |
| Capability ABI | [abi](crates/core/abi/src/lib.rs), [names](crates/core/abi/src/names.rs) | [capability set pinned to `ABI_VERSION`](crates/core/abi/tests/contract.rs) |
| Entelechy IR | [ir](crates/core/ir/src/lib.rs), [validation](crates/core/ir/src/validate.rs), [display](crates/core/ir/src/display.rs), [fusion](crates/core/ir/src/fuse.rs), [optimizer](crates/core/ir/src/optimize.rs) | [fusion bit-identity](crates/backends/cpu/tests/fusion.rs) |
| Kernel language | [language](crates/frontend/language/src/lib.rs), [parser](crates/frontend/language/src/parser.rs), [lexer](crates/frontend/language/src/lexer.rs), [scope](crates/frontend/language/src/scope.rs) | parser tests in place; [names a document sees, case by case](crates/interface/session/tests/scope.rs) |
| Compiler | [compiler](crates/frontend/compiler/src/lib.rs), [lowering](crates/frontend/compiler/src/lower/), [help](crates/frontend/compiler/src/help.rs) | [compile tests](crates/frontend/compiler/tests/compile.rs), [surface at its version](crates/frontend/compiler/tests/surface.rs), [help written as the language takes it](crates/frontend/compiler/tests/help_examples.rs), [every signature and its capability compiled](crates/frontend/compiler/tests/help_signatures.rs) |
| File writes | [writes](crates/interface/session/src/writes.rs); `else` fallback in the [evaluator](crates/core/graph/src/evaluate/mod.rs) | [session](crates/interface/session/tests/writes.rs), [lowering](crates/frontend/scene/src/lib.rs), [save replay](examples/data/tests/save.replay) |
| Laws a world writes | output `to files` into its law's document; [history origin](crates/interface/session/src/history.rs), [kept store](crates/interface/session/src/store.rs), replay `allow`; [genesis package](packages/genesis.entpkg/kernels/genesis.ent) | [genesis tests](crates/interface/session/tests/genesis.rs), [genesis replay](examples/genesis/tests/genesis.replay) |
| Shared rules | [rules](crates/core/rules/src/lib.rs), [`Backend`](crates/core/rules/src/backend.rs), [reduce tree](crates/core/rules/src/reduce.rs), [rasterizer](crates/core/rules/src/raster.rs) | every backend's parity test |
| Reference backend | [cpu](crates/backends/cpu/src/lib.rs), [word code](crates/backends/cpu/src/eval/) | [execution](crates/backends/cpu/tests/execute.rs), [optimizer](crates/backends/cpu/tests/optimizer.rs) |
| JIT backend | [jit](crates/backends/jit/src/lib.rs), [Cranelift emission](crates/backends/jit/src/codegen.rs), [WebAssembly emission](crates/backends/jit/src/wasm) | [parity](crates/backends/jit/tests/parity.rs), [matrix](crates/backends/jit/tests/matrix.rs), [as WebAssembly in node](crates/interface/entelechy/tests/wasm.rs) |
| GPU backend | [gpu](crates/backends/gpu/src/lib.rs), [WGSL](crates/backends/gpu/src/wgsl/mod.rs), [rasterizer](crates/backends/gpu/src/raster.rs), [presenter](crates/backends/gpu/src/present.rs), [wait](crates/backends/gpu/src/wait.rs) | [parity](crates/backends/gpu/tests/parity.rs), [matrix](crates/backends/gpu/tests/matrix.rs) (skip without an adapter), [shader names](crates/backends/gpu/tests/names.rs) |
| Conformance matrix | [matrix](crates/backends/conformance/src/lib.rs) | bit for bit [on the JIT](crates/backends/jit/tests/matrix.rs), ints exact and floats to rounding [on the GPU](crates/backends/gpu/tests/matrix.rs) |
| Kernel Runtime | [runtime](crates/core/kernel/src/lib.rs), [off-frame runs](crates/core/kernel/src/later.rs), [clock](crates/core/kernel/src/clock.rs) | [hot swap](crates/interface/session/tests/hot_swap.rs), [async](crates/interface/session/tests/later.rs), [mixed backends](crates/interface/session/tests/mixed.rs) |
| Scene documents | [scene](crates/frontend/scene/src/lib.rs), [domain](crates/frontend/scene/src/domain.rs) | lowering tests in place, [every form lowered](crates/frontend/scene/tests/surface.rs) |
| Graph Runtime | [graph](crates/core/graph/src/lib.rs), [evaluator](crates/core/graph/src/evaluate/mod.rs), [domain](crates/core/graph/src/domain.rs) | [composition tests](crates/interface/session/tests/composition.rs) |
| Session | [session](crates/interface/session/src/session.rs): [scene](crates/interface/session/src/scene.rs), [kernels](crates/interface/session/src/kernels.rs), [inputs](crates/interface/session/src/inputs/mod.rs), [sources](crates/interface/session/src/sources.rs), [observe](crates/interface/session/src/observe.rs), [store](crates/interface/session/src/store.rs), [bench](crates/interface/session/src/bench.rs); [formats](crates/interface/formats/src/lib.rs) | [session tests](crates/interface/entelechy/tests/session.rs) |
| Creation protocol | [protocol](crates/interface/protocol/src/lib.rs): [inspect](crates/interface/protocol/src/inspect.rs), [create](crates/interface/protocol/src/create.rs), [run](crates/interface/protocol/src/run.rs), [fields](crates/interface/protocol/src/fields.rs), [wire](crates/interface/protocol/src/wire.rs), [transport](crates/interface/protocol/src/transport.rs), [WebSocket](crates/interface/protocol/src/websocket.rs) | [client](crates/interface/entelechy/tests/protocol.rs), [live agents](crates/interface/entelechy/tests/agent.rs), [what agents found wanting](crates/interface/entelechy/tests/findings/) |
| Command line | [entelechy](crates/interface/entelechy/src/main.rs), [replay](crates/interface/session/src/replay.rs), [render](crates/interface/entelechy/src/render.rs), [packages](crates/interface/entelechy/src/package.rs), [scaffold](crates/interface/entelechy/src/scaffold.rs) | every example replay on three backends |
| C ABI | [ffi](crates/interface/ffi/src/lib.rs), [header](crates/interface/ffi/include/entelechy.h) | [ABI](crates/interface/ffi/tests/abi.rs), [header held to the exports](crates/interface/ffi/tests/header.rs), [C example](bridges/c/hello.c) |
| Player | [play](crates/interface/session/src/play/mod.rs), shared by every front; window [loop](crates/interface/player/src/main.rs), [audio](crates/interface/player/src/audio.rs), [gamepad](crates/interface/player/src/gamepad.rs); page [player.js](bridges/web/player.js) over [`entelechy_play_*`](crates/interface/ffi/src/play.rs) | [play tests](crates/interface/session/tests/play.rs), [ABI test](crates/interface/ffi/tests/abi.rs) |
| Import and export | [import](crates/interface/import/src/), [export](crates/interface/export/src/) | [MIDI](crates/interface/formats/tests/midi.rs), [round trip](crates/interface/export/tests/round_trip.rs), [SVG](crates/interface/import/src/svg.rs), [vector replays](examples/vector/tests/) |
| Bridges | [bridges](bridges/) | [bridge tests](crates/interface/entelechy/tests/bridges.rs), when the program is present |

## Libraries

Packages built from existing modes, with no engine feature of their own.

| Library | Built from | Evidence |
| --- | --- | --- |
| [Stream compaction](packages/compact.entpkg/kernels/compact.ent) | `prefix with sum`, reduce, scatter, `value(port)` | [culled replay](examples/culled/tests/culled.replay) |
| [Radix sort](packages/sort.entpkg/kernels/sort.ent) | prefix and scatter | [neighbours replay](examples/neighbours/tests/neighbours.replay) |
| [Mip levels](packages/image.entpkg/graphs/mip.esn) | `times`, `carry`, the rasterizer's `uv` | [mipmap replay](examples/mipmap/tests/ground.replay) |
| [Text](packages/text.entpkg/kernels/text.ent) | `Rows<Glyph>`, a `Grid<Float>` atlas | [text replay](examples/text/tests/text.replay) |
| [Skinning](packages/skin.entpkg/kernels/skin.ent), [rig contract](docs/RIGS.md) | maps over rows, `<~` | [rig tests](crates/interface/entelechy/tests/rigs.rs); [skin](examples/skin/tests/skin.replay), [crowd](examples/crowd/tests/crowd.replay), [reach](examples/reach/tests/reach.replay), [elbow](examples/reach/tests/elbow.replay), [stride](examples/stride/tests/stride.replay) replays |
| [Rasterization](packages/raster.entpkg/kernels/raster.ent) | scatter `min` of `Long` keys with `emit` | [edges](examples/mesh/tests/edges.replay) and [cutout](examples/mesh/tests/cutout.replay) replays on three backends, [session test](crates/interface/entelechy/tests/session.rs); [ten million triangles](examples/mesh/tests/ten_million.esn) for `bench` |
| [Vector drawing](packages/vector.entpkg/kernels/vector.ent) | `prefix with sum`, scatter `max` with `emit`, `value(port)` | [vector replays](examples/vector/tests/) on three backends |
| [Rigid bodies](packages/physics.entpkg/kernels/physics.ent), [decision](docs/DECISIONS.md#contacts-are-solved-all-at-once-each-against-its-own-share-of-a-body) | the sort package, scatter `sum` of pushes, `times` with two `carry`s, `<~` | [physics replays](examples/physics/tests/) on three backends |
| [Widgets and letters](packages/ui.entpkg/package.esn) | maps over grids, `<~` navigation state, `value(port)` and `count(port)` extents | [ui replay](examples/ui/tests/ui.replay) on three backends |
| [Font import](crates/interface/import/src/font.rs) | glyph outlines as `Rows` of segments | [importer tests](crates/interface/import/src/font.rs), on a font the tests build from the format's tables |
| [Image](packages/image.entpkg/package.esn) | maps, scatter `sum` and prefix `sum` (histograms), `times` with a carried step (jump flooding) | [image replay](examples/darkroom/tests/image.replay) on three backends |
| [Noise](packages/noise.entpkg/kernels/noise.ent) | functions over `Int` bits (`mul_wrap`, `>>>`) | [noise replay](examples/nebula/tests/noise.replay) on three backends |
| [Camera finish](packages/image.entpkg/graphs/finish.esn) | maps over grids, `size(port)` domains, the noise package | [nebula replay](examples/nebula/tests/nebula.replay) on three backends |
| [Ways over a grid](packages/grid.entpkg/kernels/paths.ent) | `times` with a carried pass over a cost grid | [skirmish replay](examples/slg/tests/skirmish.replay) on three backends |
| [Synthesis](packages/synth.entpkg/kernels/synth.ent) | a clock carried by `<~`, `rate()`, functions of time | [skirmish replay](examples/slg/tests/skirmish.replay) on three backends |

## Where each contract is enforced

### Resource and residency

- One resource ([decision](docs/TYPED-FIELD-ARCHITECTURE.md)): a Field of
  words ([host](crates/core/host/src/lib.rs)); bindings checked by shape
  and layout ([bind](crates/core/graph/src/evaluate/bind.rs)).
- `Double` and `Long`: two words, low first
  ([schema](crates/core/host/src/lib.rs), [word code](crates/backends/cpu/src/eval/code.rs),
  [JIT](crates/backends/jit/src/codegen.rs)); an unwidthed number adapted
  by [`adapt`](crates/frontend/compiler/src/lower/mod.rs), an outside value
  by `Number::as_kind` ([abi](crates/core/abi/src/lib.rs)). Held by
  [values against Rust's](crates/backends/cpu/tests/wide.rs), the matrix,
  and the [precision part](examples/data/README.md).
- Narrow storage (`as unorm8`): one rule in the Host
  ([encoding](crates/core/host/src/encoding.rs)), called by the interpreter,
  emitted by the JIT and the GPU; only a map or scan writes one, and a
  narrow link is not fused ([validator](crates/core/ir/src/validate.rs),
  [planner](crates/core/graph/src/evaluate/plan.rs)); `values()` are a
  Kernel's words, `storage()` the Field's bytes. Held by the matrix's
  `narrow` rows and the [packed part](examples/data/README.md).
- Residency: the Host owns every Field; words return from the device only
  at observation, through `Runtime::image`
  ([runtime](crates/core/kernel/src/lib.rs), [parity](crates/backends/gpu/tests/parity.rs)).
  A resident output is presented by a blit on the device
  ([presenter](crates/backends/gpu/src/present.rs)).
- Pages: a Field made from another records the words it was made from,
  generations back, and the pages it changed since each; a released
  buffer keeps the same record; a scan memory is copied, compared and
  uploaded on those pages alone
  ([host](crates/core/host/src/lib.rs), [jit](crates/backends/jit/src/lib.rs),
  [gpu](crates/backends/gpu/src/stage.rs),
  [jit parity](crates/backends/jit/tests/parity.rs),
  [gpu parity](crates/backends/gpu/tests/parity.rs)).

### Execution modes

Each mode's determinism rule is declared in the
[IR](crates/core/ir/src/lib.rs), checked by the
[validator](crates/core/ir/src/validate.rs) and
[declaration](crates/frontend/compiler/src/lower/declare.rs), and held by
the [conformance matrix](crates/backends/conformance/src/lib.rs).

| Mode | Rule | Where |
| --- | --- | --- |
| map | per element; the one mode with several outputs | [lowering](crates/frontend/compiler/src/lower/mod.rs), [WGSL](crates/backends/gpu/src/wgsl/map.rs) |
| reduce (`merge`) | a fixed pairwise tree whatever the thread count | [reduce](crates/core/rules/src/reduce.rs), [JIT parity](crates/backends/jit/tests/parity.rs) |
| scan (`scan with`) | element order; memory copied, committed only on success | [rules](crates/core/rules/src/lib.rs) |
| scatter (`scatter into … with`) | as if applied in source order; outside addresses drop | [WGSL](crates/backends/gpu/src/wgsl/scatter.rs), [parity](crates/backends/gpu/tests/parity.rs) |
| prefix (`prefix with`) | integer `sum`, `min`, `max` only | [block](crates/core/rules/src/lib.rs), [JIT](crates/backends/jit/src/lib.rs) |
| group (`group N`) | lanes meet at `exchange`, reached alike; re-run to each exchange off the device | [check](crates/core/ir/src/group.rs), [interpreter](crates/backends/cpu/src/lib.rs), [JIT parity](crates/backends/jit/tests/parity.rs), [GPU parity](crates/backends/gpu/tests/parity.rs) |

A GPU session hands scans and every module the GPU does not take
([`Backend::takes`](crates/core/rules/src/backend.rs)) to the JIT; inputs
beyond the device's bindings are packed into one buffer, copied before
each run, rows before grids and volumes
([packing](crates/backends/gpu/src/wgsl/mod.rs)). Where each node ran and
why is `Runtime::placement` ([routed test](crates/interface/entelechy/tests/session.rs),
[mixed](crates/interface/session/tests/mixed.rs), [few](crates/interface/session/tests/few.rs)). A Kernel the device's own
compiler fails on is declined from then on and runs on the JIT, said as
any routing is ([GPU `declines`](crates/backends/gpu/src/lib.rs),
`prepare_on` in [the runtime](crates/core/kernel/src/lib.rs),
[unbuilt](crates/interface/session/tests/unbuilt.rs)). The device
builds a Kernel's pipelines when it is needed: a session's graph's
Kernels as they publish, a replacement before it is ready where the
version it replaces ran there, anything else at its first run
([`Lazy`](crates/backends/gpu/src/prepare.rs),
[lazy](crates/interface/session/tests/lazy.rs)); a changed library
document is read and compiled off the frames where replacements prepare
there ([compile ahead](crates/interface/session/tests/compile_ahead.rs));
`entelechy check` says a
Kernel is large by the words it carries through its branches
([`carried`](crates/core/ir/src/size.rs)), and which published Kernels
no node uses ([large](crates/interface/session/tests/large.rs)). A `Long` on the device
avoids `select` ([prelude](crates/backends/gpu/src/wgsl/prelude.rs)).

### Fusion, repetition, native operations

- Fusion: [`fuse`](crates/core/ir/src/fuse.rs), `fuse_outputs`,
  `fuse_indexed` (a row read at an offset) and `counted` (an input only
  counted), [planner](crates/core/graph/src/evaluate/plan.rs);
  bit-identical ([test](crates/backends/cpu/tests/fusion.rs), on every
  backend [offset](crates/interface/session/tests/offset.rs)). A reduce,
  scan, scatter, prefix, group, async or native node is never fused.
- One Field per declared output or the execution fails
  ([`held_to`](crates/core/kernel/src/lib.rs), [test](crates/interface/session/tests/hot_swap.rs)).
- Repetition: one Kernel over its own output is one group, one GPU
  encoder ([`Backend::execute_repeated`](crates/core/rules/src/backend.rs)).
- Native operations are versions at generation zero
  ([`NativeOp`](crates/core/rules/src/backend.rs)). The rasterizer skips a
  triangle with any `w <= 0` whole
  ([rules](crates/core/rules/src/raster.rs), [near-plane replay](examples/mesh/tests/near.replay));
  its ports are read by field ([native replay](examples/mesh/tests/native.replay)).

### Graph, state and composition

- State: `<~` reads the last commit; an evaluation commits whole or not
  at all; a republished Kernel inherits the state
  ([state](crates/core/graph/src/evaluate/state.rs),
  [composition](crates/interface/session/tests/composition.rs)).
- Migration: `reconcile_state` runs the named `migrate` Kernel on a shape
  mismatch, else restarts the port ([an agent changing it](crates/interface/entelechy/tests/protocol.rs)).
- `times = N` expands into `name/1..N` in [scene lowering](crates/frontend/scene/src/lib.rs).
- `evaluate_for` runs only what the named nodes need.
- A named run takes, frame by frame, what the same frame of its last run
  computed under an equal key ([recall](crates/core/graph/src/evaluate/recall.rs),
  [recall test](crates/interface/session/tests/recall.rs)).
- `use` across files and cycles ([sources](crates/interface/session/src/sources.rs));
  the names each document sees, shadowing and duplicate names
  ([scope](crates/frontend/language/src/scope.rs), [session test](crates/interface/entelechy/tests/session.rs),
  [the contract case by case](crates/interface/session/tests/scope.rs)).
- Parameters of any value type, checked by layout
  ([`Param`, `value_as`](crates/core/ir/src/lib.rs), [`Value`](crates/core/abi/src/lib.rs),
  [`set_param`](crates/interface/session/src/scene.rs),
  [bind](crates/core/graph/src/evaluate/bind.rs),
  [GPU slots](crates/backends/gpu/src/wgsl/storage.rs); [cube](examples/image/README.md)).

### Domain

- A node's `domain` expression: [graph](crates/core/graph/src/domain.rs),
  [scene](crates/frontend/scene/src/domain.rs); printed back so documents
  round-trip.
- `value(port)` reads a row of scalars, bounded by `VALUE_MAX_ELEMENTS`
  ([`domain_values`](crates/core/graph/src/evaluate/bind.rs)).
- Window size and scale are part of a node's cache key.

### The language

- Sugar the parser and lowering write out
  ([parser](crates/frontend/language/src/parser.rs), [lowering](crates/frontend/compiler/src/lower/mod.rs)).
- Formula builtins (`dot`, `normalize`, `inverse`, …) are lowered to
  arithmetic, so every backend computes one formula
  ([composite](crates/frontend/compiler/src/lower/composite.rs), [matrix](crates/frontend/compiler/src/lower/matrix.rs),
  [WGSL `select`](crates/backends/gpu/src/wgsl/emitter.rs)).
- Work bounds ([decision](docs/DECISIONS.md)): `Manifest::budget`
  ([ir](crates/core/ir/src/lib.rs)), `Host::work` ([host](crates/core/host/src/lib.rs)),
  `Run` ([rules](crates/core/rules/src/lib.rs)), `Op::Spend` and `Op::Look`
  placed by [`account`](crates/core/ir/src/account.rs) in `prepare_on`
  ([runtime](crates/core/kernel/src/lib.rs)); held by the
  [IR tests](crates/frontend/compiler/tests/compile.rs) and
  [session tests](crates/interface/entelechy/tests/session.rs).
- Integer failures are the reference's everywhere: the GPU's checked
  arithmetic ([`can_fail`](crates/backends/gpu/src/wgsl/mod.rs)) and
  [`Check`](crates/core/rules/src/backend.rs)
  ([device test](crates/backends/gpu/tests/parity.rs)).
- Only nodes declaring `time.read` or `input.read` re-execute on a clock
  or input change ([composition](crates/interface/session/tests/composition.rs)).
- `entelechy help` is generated from the typing rule and the lowering
  ([help](crates/frontend/compiler/src/help.rs)); unknown names answer with
  the near miss ([names](crates/core/abi/src/names.rs)).

### Context, files and interfaces

- Input, pad, window and sound rate are context ([ABI](crates/core/abi/src/lib.rs));
  the play presses keys from a pad ([pad](crates/interface/session/src/play/pad.rs)).
- The audio callback only drains a lock-free ring
  ([audio](crates/interface/player/src/audio.rs)).
- Formats read live ([formats](crates/interface/formats/src/lib.rs)), each
  against reference vectors: [png](crates/interface/formats/src/png.rs),
  [wav](crates/interface/formats/src/wav.rs), [inflate](crates/interface/formats/src/inflate.rs),
  [midi](crates/interface/formats/src/midi.rs); data tables, CSV and JSON
  read as the reading Kernel lays them out ([table](crates/interface/formats/src/table.rs),
  [table tests](crates/interface/session/tests/tables.rs)).
- The protocol is an interface over the session: `prepare` never
  publishes, `publish` checks a generation, `compose` rewrites the
  document ([protocol](crates/interface/protocol/src/lib.rs)).
- The C ABI's lifetimes are in its [header](crates/interface/ffi/include/entelechy.h).
- Application tests are `.replay` data ([replay](crates/interface/session/src/replay.rs)),
  run by the verification in [HANDOFF](HANDOFF.md).

### Runtime and backends

- The backends are peers behind [`Backend`](crates/core/rules/src/backend.rs);
  the runtime takes validated IR and the backend the session picks
  ([`runtime_for`](crates/interface/session/src/session.rs)).
- `prepare` does the work, `publish` swaps a pointer; `prepare_later`
  moves it to a worker ([hot swap](crates/interface/session/tests/hot_swap.rs)).
- A sync compiles every new or changed Kernel of the documents it read,
  and prepares the new ones, on every core at once through a
  [`Preparer`](crates/core/kernel/src/lib.rs); each document then
  publishes in order as if alone ([prime](crates/interface/session/tests/prime.rs)).
  `entelechy check` checks the documents a scene uses the same way, each on
  its own, and says what they found in their order
  ([spread](crates/interface/session/src/spread.rs)).
- The interpreter is the reference; the first failing element in
  row-major order is reported ([compile](crates/backends/cpu/src/eval/compile.rs),
  [execute](crates/backends/cpu/tests/execute.rs)).
- Async runs are adopted when they land ([later](crates/core/kernel/src/later.rs)).
- `trace` runs one element of the unoptimized Kernel up to its failure,
  which the compiler's value places put in the source
  ([trace](crates/backends/cpu/src/lib.rs),
  [places](crates/frontend/compiler/src/lib.rs),
  [test](crates/interface/entelechy/tests/session.rs)).
- The optimizer rewrites exactly and moves nothing that can fail
  ([optimize](crates/core/ir/src/optimize.rs), [test](crates/backends/cpu/tests/optimizer.rs)).
- A platform without threads, links, native code or a clock does the same
  work another way ([decision](docs/DECISIONS.md), [later](crates/core/kernel/src/later.rs),
  [store](crates/interface/session/src/store.rs), [clock](crates/core/kernel/src/clock.rs),
  [bridges test](crates/interface/entelechy/tests/bridges.rs)).

### Measuring

- A profile line subtracts `Backend::measurement_floor`
  ([runtime](crates/core/kernel/src/lib.rs), [bench](crates/interface/session/src/bench.rs)).
- The GPU buffer pile serves the smallest buffer at most twice the size
  ([`Spare::take`](crates/backends/gpu/src/lib.rs)).
- Handle liveness and node lookup are kept counts, checked against the
  walk in debug ([ledger](crates/core/graph/src/evaluate/ledger.rs),
  [`Graph::index`](crates/core/graph/src/lib.rs)).

## Device facts

Each is a workaround guarded by a test that fails without it; remove one
only after running that test on the device.

- Metal hoists a scan step's first memory load out of the step loop: a
  `storageBarrier()` sits between steps ([parity](crates/backends/gpu/tests/parity.rs)).
- One dispatch dimension holds 65,535 workgroups: long `Rows` dispatch as
  a grid ([wgsl](crates/backends/gpu/src/wgsl/mod.rs)).
- Metal narrows a sixteen-bit division by a constant wrongly: every `/`
  and `%` multiplies in `OPAQUE_ONE` (matrix row `narrow division`).
- `u32 / u32` is wrong with the top bit set: `udiv` and `urem` divide
  halved dividends signed.
- WGSL's float `%` rounds the quotient: a Float remainder is an `fmod` on
  a fused multiply-add.
- Fast math may fold `inf * 0`: a NaN made on the GPU is not promised.
- Narrow outputs sharing a word are `atomic<u32>` stored by `atomicOr`,
  with integer encodings rather than `pack2x16float`
  ([wgsl](crates/backends/gpu/src/wgsl/storage.rs) `narrow_store_body`).
- wgpu's error scopes are thread-local only with its `std` feature, which
  the GPU backend enables.
- Uniforms are rewritten per node with `queue.write_buffer`, which orders
  against submissions: a frame is not one submission.

## Declined

Each stays out until a new measurement reopens it.

- A node-level parallel scheduler: a map uses every core; `async` covers
  work off the frame.
- A `times` that stops on a residual: a step count is its author's budget.
- `break` in hot loops: the showcase keeps `until`.
- A sixteen-wide reduction fold on the CPU: a dependency chain.
