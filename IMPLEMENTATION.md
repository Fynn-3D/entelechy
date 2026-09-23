# Implementation evidence

**Answers:** where the code is, and which test establishes each contract.

Every line here points at code or at a test. Follow the link before repeating
what it says; the test is the claim, this is the index to it.

## The four proofs

| Proof | Where it is exercised |
| --- | --- |
| `.orla` compiles | [compiler tests](crates/compiler/tests/compile.rs) |
| Kernel calls Host only through ABI | [scope tests](crates/abi/src/lib.rs), [denied execution](crates/cpu/tests/execute.rs), [validator capability checks](crates/ir/src/validate.rs) |
| Kernel hot-swaps without restart | [hot swap tests](crates/runtime/tests/hot_swap.rs), [session tests](crates/cli/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/graph/tests/composition.rs), [example scene](examples/grayscale/scene.osn) |

## Layers

| Layer | Implementation | Evidence |
| --- | --- | --- |
| Host Core | [host](crates/host/src/lib.rs), [files](crates/host/src/file.rs) | handles, budgets, bounds |
| Capability ABI | [abi](crates/abi/src/lib.rs) | grants checked before every Host call |
| Orla IR | [ir](crates/ir/src/lib.rs), [validation](crates/ir/src/validate.rs), [display](crates/ir/src/display.rs), [fusion](crates/ir/src/fuse.rs), [optimizer](crates/ir/src/optimize.rs) | typing, SSA order, capability coverage; [fusion bit-identity](crates/ir/tests/fusion.rs) |
| Kernel language | [language](crates/language/src/lib.rs), [parser](crates/language/src/parser.rs), [lexer](crates/language/src/lexer.rs) | syntax, precedence, positioned errors, near misses |
| Compiler | [compiler](crates/compiler/src/lib.rs), [lowering](crates/compiler/src/lower/), [help](crates/compiler/src/help.rs) | [compile tests](crates/compiler/tests/compile.rs) |
| Shared rules | [rules](crates/rules/src/lib.rs), [reduce tree](crates/rules/src/reduce.rs), [matrix formulas](crates/rules/src/matrix.rs), [rasterizer contract](crates/rules/src/raster.rs) | every backend's parity test runs by them |
| Reference backend | [cpu](crates/cpu/src/lib.rs), [word code](crates/cpu/src/eval/) | [execution tests](crates/cpu/tests/execute.rs), [optimizer tests](crates/cpu/tests/optimizer.rs) |
| JIT backend | [jit](crates/jit/src/lib.rs), [Cranelift emission](crates/jit/src/codegen.rs) | [parity](crates/jit/tests/parity.rs), [matrix](crates/jit/tests/matrix.rs), every replay on `jit` |
| GPU backend | [gpu](crates/gpu/src/lib.rs), [WGSL emission](crates/gpu/src/wgsl.rs), [rasterizer](crates/gpu/src/raster.rs), [presenter](crates/gpu/src/present.rs) | [parity](crates/gpu/tests/parity.rs), [matrix](crates/gpu/tests/matrix.rs); both skip without an adapter |
| Kernel Runtime | [runtime](crates/runtime/src/lib.rs), [off-frame runs](crates/runtime/src/later.rs) | [hot swap](crates/runtime/tests/hot_swap.rs), [async](crates/runtime/tests/later.rs), [mixed clocks](crates/runtime/tests/mixed.rs) |
| Graph Runtime | [graph](crates/graph/src/lib.rs), [document](crates/graph/src/document.rs), [evaluator](crates/graph/src/evaluate.rs), [domain](crates/graph/src/domain.rs) | [composition tests](crates/graph/tests/composition.rs) |
| Interface | [cli](crates/cli/src/main.rs), [session](crates/cli/src/session.rs), [replay](crates/cli/src/replay.rs), [bench](crates/cli/src/bench.rs), [render](crates/cli/src/render.rs) | [session tests](crates/cli/tests/session.rs), every example replay on three backends |
| Creation protocol | [protocol](crates/cli/src/protocol.rs), [contract](docs/PROTOCOL.md) | [client test](crates/cli/tests/protocol.rs); [an agent's record](examples/scanline/agent.jsonl) replayed |
| C ABI | [ffi](crates/ffi/src/lib.rs), [header](crates/ffi/include/orla.h) | [ABI tests](crates/ffi/tests/abi.rs), [C example](bridges/c/hello.c) |
| Player | [window loop](crates/player/src/main.rs), [audio](crates/player/src/audio.rs), [gamepad](crates/player/src/gamepad.rs) | presentation and blit tests; [material ball](examples/material-ball/README.md) |
| Import and export | [import](crates/import/src/), [export](crates/export/src/) | [MIDI tests](crates/import/tests/midi.rs), the [score](examples/score/README.md) and [clip](examples/clip/README.md) replays |
| Feature matrix | [matrix](crates/matrix/src/lib.rs) | [on the JIT](crates/jit/tests/matrix.rs) bit for bit, [on the GPU](crates/gpu/tests/matrix.rs) ints exact and floats to rounding |

## Capabilities

| Capability | Where it lives | What establishes it |
| --- | --- | --- |
| Five execution modes: map, reduce, scan, scatter, prefix | [`Mode`](crates/ir/src/lib.rs), [declaration](crates/compiler/src/lower/declare.rs) | the matrix's six mode wrappers; the contracts below |
| The language's own features | [contract](docs/DOCUMENTS.md), [parser](crates/language/src/parser.rs), [lowering](crates/compiler/src/lower/) | 35 feature rows of the matrix, on two backends against the reference |
| Optimizer rewrites: inlining, hoisting, folding, narrowing, running a literal loop | [optimize](crates/ir/src/optimize.rs) | [every matrix case word for word against the unoptimized IR](crates/cpu/tests/optimizer.rs) |
| Fusion, and repetition where fusion refuses | [fuse](crates/ir/src/fuse.rs), [planner](crates/graph/src/evaluate.rs) | [fusion tests](crates/ir/tests/fusion.rs), [composition](crates/graph/tests/composition.rs) |
| Stream compaction: which ones, not just how many | [compact package](packages/compact.orlapkg/kernels/compact.orla) | [culled replay](examples/culled/tests/culled.replay) — the same picture with the culling on and off |
| A domain a Kernel decides | [`value(port)`](crates/graph/src/domain.rs), [binding](crates/graph/src/evaluate.rs) `domain_values` | [decided](examples/decided/tests/decided.replay) and [surface](examples/surface/tests/surface.replay) replays |
| Rasterizing: the near plane, the cutout, the contract | [rules](crates/rules/src/raster.rs), [reference](crates/cpu/src/raster.rs), [device](crates/gpu/src/raster.rs) | [near-plane replay](examples/mesh/tests/near.replay), [parity](crates/gpu/tests/parity.rs) |
| A rig: a skeleton, its clips, and the mesh that follows | [skin package](packages/skin.orlapkg/kernels/skin.orla), [glTF skins and animations](crates/import/src/gltf.rs), [the Blender bridge's push](bridges/blender/orla_bridge.py) | [import tests](crates/import/src/gltf.rs) (a hand-written rig: parents, inverse binds, a slerped clip), [skin replay](examples/skin/tests/skin.replay) on three backends, and [the two ways in posing alike](crates/cli/tests/bridges.rs) |
| Mip levels as a library, not a primitive | [picture package](packages/picture.orlapkg/kernels/mip.orla), [strip](packages/picture.orlapkg/graphs/mip.osn) | [mipmap replay](examples/mipmap/tests/ground.replay) |
| Async nodes | [off-frame runs](crates/runtime/src/later.rs), [evaluator](crates/graph/src/evaluate.rs) | [runtime tests](crates/runtime/tests/later.rs), [async replay](examples/async/tests/async.replay) |
| Packages and versions | [packages](packages/README.md), [pack and unpack](crates/cli/src/package.rs), [by name](crates/cli/src/session.rs) `in_package` | [session tests](crates/cli/tests/session.rs), [package replay](examples/package/tests/package.replay) |
| Files in and out: PNG, PPM, PGM, WAV, glTF, MIDI, bytes, frames, text | [host files](crates/host/src/file.rs), [png](crates/cli/src/png.rs), [wav](crates/cli/src/wav.rs), [import](crates/import/src/), [export](crates/export/src/) | [session tests](crates/cli/tests/session.rs), the [level](examples/level/tests/level.replay), [clip](examples/clip/tests/clip.replay) and [score](examples/score/tests/score.replay) replays |
| Measuring a scene, and a profile that adds up | [bench](crates/cli/src/bench.rs), [runtime](crates/runtime/src/lib.rs) `last_measured`, `measurement_floor`, [device clock](crates/gpu/src/lib.rs) | [session tests](crates/cli/tests/session.rs), [protocol tests](crates/cli/tests/protocol.rs), [mixed clocks](crates/runtime/tests/mixed.rs) |
| A scene to start from | [scaffold](crates/cli/src/scaffold.rs) | [session tests](crates/cli/tests/session.rs): made, checked, replayed |
| Bridges to other programs | [bridges](bridges/) in each program's own language | [bridge tests](crates/cli/tests/bridges.rs), run when node, Chrome or Godot is present |

## Contracts worth knowing

### Resource and residency

- There is one resource, following [the Typed Field decision](Orla%20Typed%20Field%20Architecture.md):
  a Field of elements in one of three shapes, `Rows`, `Grid`, or `Volume`,
  with one 32-bit word per scalar ([host](crates/host/src/lib.rs)). Port
  types are structural (`element`, `shape`, `schema`); `Image` is the
  compiler's alias for `Grid<Vec4>` and `type` declares more aliases
  ([compiler](crates/compiler/src/lower/mod.rs)). Both backends carry every
  element as words, so integers and booleans are exact and the two agree
  bit for bit on the examples ([execution test](crates/cpu/tests/execute.rs),
  [parity test](crates/gpu/tests/parity.rs)); the GPU dispatches over three
  axes and keeps every shape resident. Bindings are checked by shape and
  layout, never by spelling ([evaluator](crates/graph/src/evaluate.rs)).
  The [particles](examples/particles/README.md) (rows),
  [tiles](examples/tiles/README.md) (a grid of structs through an alias),
  [Life](examples/life/README.md) (a grid of booleans and integers), and
  [volume](examples/volume/README.md) examples are the working proofs.

- Residency: the Host owns every image, but a slot may hold its pixels on a
  device as an opaque object the backend understands
  ([host](crates/host/src/lib.rs)). The GPU backend binds resident inputs
  directly, uploads Host-memory inputs once, and leaves outputs resident;
  committed state therefore ping-pongs on the device. Pixels return only at
  observation boundaries through [`Runtime::image`](crates/runtime/src/lib.rs)
  (export, protocol `read`, replay `expect`, the Player's present), which is
  where device work is waited for ([residency test](crates/gpu/tests/parity.rs)).

- Presentation is an observation the GPU can do itself: the Player gives the
  GPU backend its window, and a resident output is drawn onto the surface by
  a blit shader that reads the image's storage buffer; the frame never
  crosses to the Host. The title reports the path taken and the download
  count. Without a GPU, or when the surface cannot be configured, the Player
  downloads and blits through softbuffer ([presenter](crates/gpu/src/present.rs)).

- Sampling authority is a builtin gated by layout, not by the word
  `Image`: `at_linear(f, position)` interpolates between neighbouring
  elements of any input whose layout is all floats (a struct of floats
  included, mixed fieldwise), bilinear over grids and trilinear over
  volumes, with element centres at half-integers and clamped edges. The
  CPU sums weighted corners in words; the GPU emits one helper per input
  and one mixer per float struct ([interpreter](crates/cpu/src/eval/compile.rs),
  [WGSL](crates/gpu/src/wgsl.rs), [parity test](crates/gpu/tests/parity.rs)).
  The [volume example](examples/volume/README.md) reads its cloud that way.

### Execution modes

- Reduction is the second execution mode, declared by a `merge(a, b)`
  block: the entry maps over the first input's domain and `merge`
  folds the results pairwise into one `Rows` element; the manifest
  carries `mode`, the module carries `merge`, and the validator checks
  the fold's signature against the output element
  ([IR](crates/ir/src/lib.rs), [validator](crates/ir/src/validate.rs),
  [compiler](crates/compiler/src/lower/mod.rs)). The CPU folds level by level
  in a fixed tree regardless of thread count; the GPU maps into a scratch
  buffer and folds with a second pipeline in the same tree, so integers
  are exact on both and floats agree to rounding
  ([CPU](crates/cpu/src/lib.rs), [WGSL](crates/gpu/src/wgsl.rs),
  [parity test](crates/gpu/tests/parity.rs)). The evaluator gives a reduce
  node the domain of one element and keeps it out of fusion
  ([evaluator](crates/graph/src/evaluate.rs)). The
  [exposure example](examples/exposure/README.md) meters a frame and
  tone-maps by the result.

- Ordered work is the third execution mode, declared by `scan with name =
  init { ... }` in place of `compute`: the entry runs once per element of
  the first input, in order, reading the previous result through the same
  `Carried` op a loop body uses (`init`, a module function, supplies the
  first) and writing every step as that element of a `Rows` output of the
  input's count. The CPU walks the rows on one thread; the GPU walks them
  in one invocation, so the two agree exactly on integers and to rounding
  on floats ([CPU](crates/cpu/src/lib.rs), [WGSL](crates/gpu/src/wgsl.rs),
  [parity test](crates/gpu/tests/parity.rs)). Scans are never fused. The
  [audio example](examples/audio/README.md) filters each block with one
  and carries the filter's memory across blocks through Graph state.

- Ordered work may own memory: `memory ram: Rows<Int>;` is a port that
  is an input and an output at once (last in both lists of the
  manifest, named in `memories`), which a scan's steps read with `at` as
  the earlier steps left it and write with `put(ram, address, value)` in
  step order, an address outside dropped; a function may `put` into a
  Field parameter, so a store is written once (it is a write of the step
  that calls it, and `init` may not call one, checked over the call
  graph). The CPU works on a copy of the
  bound Field and hands the copy back as the output; the GPU copies the
  input's buffer, binds the copy read-write, and adopts it as the output;
  the bound Field is never written, so a failed evaluation commits
  nothing, and both backends agree bit for bit
  ([IR](crates/ir/src/lib.rs), [validator](crates/ir/src/validate.rs),
  [compiler](crates/compiler/src/lower/mod.rs), [CPU](crates/cpu/src/eval/compile.rs),
  [WGSL](crates/gpu/src/wgsl.rs), [parity](crates/gpu/tests/parity.rs),
  [feedback test](crates/graph/tests/composition.rs)). With the integer
  bit operators, this is what a machine needs: random-access RAM written
  step by step, persisting across frames through `<~`; the
  [CHIP-8 example](examples/chip8/README.md) is one, every opcode, its
  replay reading the screen memory on both backends; the
  [Thumb example](examples/thumb/README.md) is a 32-bit one, the
  ARM7TDMI's 16-bit instruction set with byte-addressed RAM through
  store functions, wrapping arithmetic and the logical shift, checked
  against a reference simulator on all three backends. Two GPU facts
  came out of it: a scan step's first load of a memory element was
  hoisted out of the step loop by the Metal compiler, so the loop now
  has a `storageBarrier()` between steps
  ([regression test](crates/gpu/tests/parity.rs)); and a `Rows` map of
  more than 524,280 elements exceeded the 65,535 workgroups one
  dispatch dimension allows, so long rows are dispatched as a grid, the
  fold carried in the context ([dispatch](crates/gpu/src/wgsl.rs)).

- Writing where a Kernel computes is the fourth execution mode, declared
  by `scatter into target with rule` in place of `compute`: the entry
  maps over the first input and yields an address and a value, and the
  output is the target with every value combined into the element at its
  address, as if the sources were applied one after another in element
  order. That order is the determinism rule: `sum`, `min`, `max`, and
  `last` come out the same on both backends, floats included, and an
  address outside the target is dropped ([IR](crates/ir/src/lib.rs),
  [validator](crates/ir/src/validate.rs), [compiler](crates/compiler/src/lower/mod.rs)).
  The CPU applies the pairs in order over a copy of the target
  ([backend](crates/cpu/src/lib.rs)); the GPU stores the pairs in a
  parallel pass, copies the target on the device, and applies them by
  plan: the order-free rules (an integer `sum`, `min` and `max` on one
  total order of floats as keys, `last` as the greatest source index then
  a gather) one invocation per pair with atomics, a float `sum` by a
  bitonic sort of the pairs by target then source and one invocation per
  target folding its run in that order, the CPU's sequence exactly
  ([WGSL](crates/gpu/src/wgsl.rs), [parity test](crates/gpu/tests/parity.rs)
  with a hundred thousand sources);
  the evaluator sizes a scatter node by its target and never fuses it
  ([evaluator](crates/graph/src/evaluate.rs)). The
  [particles example](examples/particles/README.md) splats its rows onto
  a canvas this way.

- A prefix fold is the fifth execution mode, declared by `prefix with
  rule` in place of `compute`: the entry maps over the first input, a
  `Rows`, and element `i` of the `Rows` output is the entry's values at
  `0..=i` combined by the rule ([IR](crates/ir/src/lib.rs),
  [validator](crates/ir/src/validate.rs),
  [compiler](crates/compiler/src/lower/declare.rs)). The rule is named
  rather than carried because a prefix folds in whatever grouping a
  backend likes, so it must be one that gives the same words in any of
  them: an integer `sum`, `min` and `max` on the scatter's one total
  order of floats. A float `sum` is refused — it rounds in the order it
  is added, which is what `scan` is for — and so is `last`, which as a
  prefix is a map. The CPU and the JIT map and then walk the values in
  element order ([CPU](crates/cpu/src/lib.rs), [JIT](crates/jit/src/lib.rs));
  the GPU folds in three passes over blocks of about the square root of
  the count ([block](crates/rules/src/lib.rs),
  [WGSL](crates/gpu/src/wgsl.rs)) — one invocation walks each block and
  leaves its total, one walks the totals, one hands every element its
  block's offset — which needs no shared memory and no barrier, and is
  held to the reference's words by
  [the parity test](crates/gpu/tests/parity.rs) (NaN, both zeros and the
  infinities under `max`) and by a
  [matrix row](crates/matrix/src/lib.rs) under every feature.

- A scatter node may set its own `domain`: its body runs once per element
  of it, the first input's extent by default, while the output keeps the
  target's extent — so a step over a product of two counts needs no Field
  of that size. The target must exist as a Field, since it holds what the
  sums start from. A reduction's output may be written as its element
  type (`output result: Loss;` is `Rows<Loss>`).

### Fusion, repetition, native operations

- Fusion is backend-neutral and exact: a producer read by exactly one
  consumer, only through `sample`, over the same domain, that is neither a
  graph output, a state source, nor pinned, is inlined into that consumer
  by [`fuse`](crates/ir/src/fuse.rs) and executed as one detached Kernel
  version. Results are bit-identical on the CPU
  ([IR test](crates/ir/tests/fusion.rs), [graph tests](crates/graph/tests/composition.rs));
  absorbed nodes are not materialized until pinned.

- Iteration fusion is the same idea without inlining: a node placed several
  times over its own output — the same Kernel of the same generation, the
  same parameters and bindings, the same domain, Map, neither async nor
  pinned, and nothing else reading what passes between — is one group that
  repeats ([planner](crates/graph/src/evaluate.rs)). The last placement
  keeps its name and its output, as a fused consumer does; the backend is
  asked whether it can repeat itself
  ([`Backend::execute_repeated`](crates/runtime/src/lib.rs)), which the GPU
  does by recording the passes into one encoder with two buffers taking
  turns ([gpu](crates/gpu/src/lib.rs)), and otherwise the runtime runs the
  group N times and releases what is between
  ([graph test](crates/graph/tests/composition.rs)). Where the Kernel only
  `sample`s its input, ordinary fusion gets there first and inlines every
  time into one Kernel; repetition is what a Kernel that reads its
  neighbours gets instead.

- Structs that are the same are one entry in a fused module's table
  ([`fuse`](crates/ir/src/fuse.rs)): the consumer's sample of the fused
  input becomes the producer's result, so a whole element the consumer
  passes around — returned from one arm of an `if` beside one it builds
  itself — must carry the consumer's own struct. Appending the
  producer's copy beside it made the two arms yield different indices of
  the same type and the fused module did not validate. A struct may only
  contain structs before it, so one pass in order maps every producer
  struct into the fused table, reusing an identical one where there is
  one.

- Native operations are device access published as versions: the runtime
  registers each [`NativeOp`](crates/runtime/src/lib.rs) at generation
  zero with a manifest and a textual contract, refuses to publish a Kernel
  over it, and dispatches it to `Backend::execute_native`; the evaluator
  never fuses one. `native.rasterize@1` is the first: on the CPU a
  reference rasterizer ([raster](crates/cpu/src/raster.rs)), on the GPU a
  render pipeline into two float targets and a depth target compacted
  into the tight word layout ([raster](crates/gpu/src/raster.rs)); they
  agree on every interior pixel ([parity test](crates/gpu/tests/parity.rs),
  [runtime test](crates/runtime/tests/hot_swap.rs)). The contract has no
  clipping: a triangle with any vertex at `w <= 0` is skipped whole. A
  device rasterizer would clip it and draw what is left, so the vertex
  stage reads all three corners' `w` and sends the triangle to one point
  when any of them is behind the eye; a triangle of no area draws
  nothing on either side, which is also how a Kernel culls
  ([near-plane replay](examples/mesh/tests/near.replay)). Projection and
  shading stay Kernels on either side of it
  ([mesh example](examples/mesh/README.md)).

- A cutout is the rasterizer's second form, `native.rasterize@2`: one
  more input, an `Image` whose alpha says what the surface covers, and a
  `cutout` parameter. A fragment whose alpha there — read at its own uv,
  as `at_linear(mask, uv * size(mask))` reads it — is below the cutout is
  not drawn at all: it writes no surface and settles no depth, so what
  stands behind a leaf's edge is seen through it. Both backends read the
  mask by the same formula
  ([`mask_alpha`](crates/rules/src/raster.rs)), so the edge is the same
  pixel on either; the device is held to the reference through a
  checkerboard's holes ([parity test](crates/gpu/tests/parity.rs)). It is
  what foliage is made of: a leaf is two triangles and a texture, not a
  thousand triangles of leaf.

- Mip levels are a library, not a language feature: the strip of
  halvings is a sub-graph placed `levels` times with `carry`
  ([`picture.orlapkg`](packages/picture.orlapkg/graphs/mip.osn)), and the level a pixel
  asks for is read off the surface beside it — a rasterized surface
  carries `uv` at every pixel, so the neighbour's `uv` is what a
  derivative would say, and its triangle `id` says whether the
  neighbour is the same surface
  ([`picture.orlapkg`](packages/picture.orlapkg/kernels/mip.orla),
  [mipmap example](examples/mipmap/README.md)). What it does not give is
  anisotropy, which is more taps in the same library rather than
  anything the language lacks. A placement does not know which
  repetition it is, so the level being written is carried beside the
  strip as a row of one `Int`.

### Graph, state and composition

- State lives in the Graph, not in Kernels: a `port <~ node.output from
  initial` binding reads the source as committed by the previous successful
  evaluation. Every node of one evaluation reads the same committed state;
  the evaluation commits atomically at its end; a failed evaluation commits
  nothing; a republished Kernel inherits the state; `reset_state` returns to
  the initial sources ([evaluator](crates/graph/src/evaluate.rs),
  [feedback tests](crates/graph/tests/composition.rs),
  [Life example](examples/life/README.md)).

- State migration (the plan's section 24): a `<~` binding may name a
  migration Kernel, `migrate id@v`, a map from the old element layout to
  the new; before the nodes run, the evaluator compares every committed
  state's shape and layout with its port's and, on a mismatch, runs the
  migration once over the state (its input checked against the state,
  its output against the port) or, with none named, restarts the port
  from its initial source, reporting either; a migration that fails
  leaves the state as it was ([evaluator](crates/graph/src/evaluate.rs)
  `reconcile_state`, [composition test](crates/graph/tests/composition.rs),
  [an agent's record](examples/migrate/agent.jsonl)). The check is the
  layout comparison every evaluation already made; nothing is added to a
  frame that migrates nothing.

- A node repeats with `times = N;` and `carry port <- output;`: the
  document lowering expands it into `name/1..N` chained through the
  carried ports, with `name.port` meaning the last; `N` may be a graph
  param, which the session then treats as structural and lowers the
  document again when it changes ([document](crates/graph/src/document.rs),
  [tests](crates/graph/src/document.rs), [session test](crates/cli/tests/session.rs)).
  The fluid's Jacobi solve, the SLG's relaxation passes and the mesh
  example's subdivision level are written that way.

- A node may be a filled Field: `node zeros: Rows<Float> { domain = 65; }`,
  which document lowering turns into a one-parameter Kernel (`value`, the
  element's zero by default, resolved through the used documents'
  aliases), so a scene's starting values are lines rather than Kernels
  ([document](crates/graph/src/document.rs)).

- An evaluation can be demanded: `Evaluator::evaluate_for` runs the named
  nodes, what they read, every `<~` source, and never-produced outputs, and
  leaves the rest at its last value; a replay demands the picture only on
  frames a pixel expectation follows, so a replay of state costs what the
  state costs ([evaluator](crates/graph/src/evaluate.rs), [replay](crates/cli/src/replay.rs)).

- Declarations cross files through `use`: a document's used files (and
  theirs) contribute their top-level structs and type aliases to every
  Kernel the document holds, resolved by the session and `check` through
  a `declare` callback the graph loader applies to the root and each
  sub-graph ([graph](crates/graph/src/document.rs),
  [session](crates/cli/src/session.rs)). A changed library republishes
  the Kernels that inherit from it and re-reads the root; two different
  declarations of one name are refused with both files named
  ([session test](crates/cli/tests/session.rs)).

- Functions are shared like structs: declared at a document's top level
  and inherited through `use`; one with a Field parameter is a template
  the compiler instantiates per bound port, so a grid walk or a staggered
  read is written once for every Kernel that needs it
  ([lowering](crates/compiler/src/lower/mod.rs), [tests](crates/compiler/tests/compile.rs)).

- Documents: one grammar for `.orla` libraries and `.osn` graphs
  ([contract](docs/DOCUMENTS.md), [frontend](crates/language/src/lib.rs),
  [graph lowering](crates/graph/src/document.rs),
  [session loader](crates/cli/src/session.rs)). Nodes reference Kernels by
  id or carry them inline; the loader publishes inline Kernels and follows
  `use`. Proof: [parser tests](crates/language/src/parser.rs),
  [lowering tests](crates/graph/src/document.rs),
  [session tests](crates/cli/tests/session.rs), and the
  [self-contained material ball](examples/material-ball/scene.osn).

- Packages (the plan's section 43) are directories with a `package.osn`
  manifest, `package name@version { language = N; ir = N; }` plus the
  `use`s that bring its Kernels; a scene's `use` of one expands, in the
  session's use expansion, to what the manifest uses once its versions
  fit, and files are canonicalized so a Kernel reached by two paths has
  one owner. `orla pack`/`unpack` carry a package as one file in Orla's
  own container, no dependency added. Versions (section 44) are four
  constants, `ABI_VERSION`, `LANGUAGE_VERSION`, `IR_VERSION`,
  `PACKAGE_VERSION`, printed by `orla version` and in the protocol's
  `status`.

- Stream compaction is a package, not an engine feature
  ([compact](packages/compact.orlapkg/kernels/compact.orla)): a rank is
  a prefix sum over the flags, which `prefix with sum` is; the count is
  a reduce;
  and putting each kept index where its rank says is a scatter, whose
  addresses outside the target are dropped, which is what a flag of zero
  sends. With `value(count)` sizing what runs next, that is culling: a
  chunk nobody can see is never placed, not skipped. The rank was ordered
  work until the prefix mode; a device folds it in blocks now, which is
  what lets a scene rank the elements and not only the things they expand
  into — see [surface](examples/surface/README.md), whose rank is over
  two million cells. What a frame costs is then what it can see, and the
  world above that is free: [culled](examples/culled/README.md) carries
  a hundred and sixty million triangles it could never place.

- A Field comes in from outside through the protocol's `write`, which
  replaces a state port's committed value with words in its layout
  ([evaluator](crates/graph/src/evaluate.rs), [protocol](crates/cli/src/protocol.rs)).

### Domain

- A node's `domain` may be an expression over its inputs and parameters
  ([domain](crates/graph/src/domain.rs)): a small arithmetic language with
  `size(port)` and `count(port)`, accepted at document lowering, resolved
  when the node binds (and at planning, so fusion and `orla graph` see
  the extent), and printed back so documents round-trip. It lets a Kernel
  that changes topology size its output from its input, as the
  [mesh example](examples/mesh/README.md)'s subdivision does, and the
  [tiles example](examples/tiles/README.md) draw a map at its tile size.

- A node's `domain` may read a number a Kernel wrote
  ([`value(port)`](crates/graph/src/domain.rs),
  [`domain_values`](crates/graph/src/evaluate.rs)): element `i` (0 unless
  given) of the Field a port is bound to, as the `Int` or `Float` it
  holds. It is the one domain term that is data rather than shape, and
  so the only way a scene does less work when a Kernel finds there is
  less to do — a Kernel that reads a count and returns early still
  writes an element for every one of them, and the Field is what costs.
  The producer has run by the time the node binds, so the number is this
  frame's; reading it brings that Field to the Host, which on a device
  is a wait, so the port is held to a row of scalars and to
  `VALUE_MAX_ELEMENTS` of them and several nodes reading one Field wait
  once. No backend knows about any of it. A node sized this way is not
  knowable before its producer runs, so it would fall out of fusion; a
  consumer with no `domain` of its own runs over its first input's
  extent, which is the producer's whatever it turns out to be, so the
  producer's expression moves onto the fused node under the names fusion
  gives its ports (`renamed`). A shorter Field is the only thing that is
  actually shorter: [decided](examples/decided/README.md) flies over a
  sheet whose vertex count the camera's height decides.

### The language

- Structs are Kernel-internal record types: declared in `.orla`, indexed in
  [the IR struct table](crates/ir/src/lib.rs), built and updated with
  `Name { .. }` literals, and allowed wherever a value is (loops carry them,
  functions take and return them). They never cross the ABI as parameters or
  ports. The CPU backend lays them out as words, fields in order
  ([interpreter](crates/cpu/src/eval/compile.rs)); the GPU backend emits native WGSL
  structs. Proof: [struct tests](crates/cpu/tests/execute.rs),
  [diagnostics](crates/compiler/tests/compile.rs),
  [parity](crates/gpu/tests/parity.rs), and the
  [ray-march example](examples/raymarch/kernels/march.orla).

- Fixed-size arrays `[T; N]` are structs of `N` fields of one type,
  marked with their element in [the struct table](crates/ir/src/lib.rs)
  and interned by the compiler so two spellings are one type: they lay
  out, build (`[a, b, c]`, `[v; N]`) and extract (a literal index) like
  any struct on every backend, and a run-time index is one more
  operation, `subscript`, clamped into the array (the CPU copies the
  element's words, the GPU indexes a WGSL `array<T, N>` alias, the JIT
  spills the array to a stack slot and loads at the index — one slot
  per array in the block that stored it, so reading a second place out
  of an array a Kernel built is one more load and not another store of
  every word). Proof: the
  [feature matrix](crates/matrix/src/lib.rs) under every mode on both
  backends, [diagnostics](crates/compiler/tests/compile.rs), and the
  [CHIP-8](examples/chip8/kernels/chip8.orla) and
  [Thumb](examples/thumb/kernels/thumb.orla) ROMs written as data.

- A table, an array whose words are known when a Kernel compiles, costs
  nothing to build: the JIT puts its words in the module's read-only
  data once and a `subscript` loads the element at the clamped offset
  (an array picked out of a table of tables keeps its address, so a
  second subscript is one more load, not a copy); the interpreter
  folds the literals into one `Words` instruction and drops the ones
  nothing else reads, a named one kept for its trace
  ([JIT](crates/jit/src/codegen.rs),
  [interpreter](crates/cpu/src/eval/compile.rs)).

- Matrices `Mat2/3/4` are value types of the ABI, column-major floats,
  with the operator table in [the IR](crates/ir/src/lib.rs) (`Mat * Mat`,
  `Mat * Vec`, `Vec * Mat`, scaling, `+ -`, `==`) and three intrinsics
  (`transpose`, `determinant`, `inverse`). The formulas live once, in
  [`orla_rules::matrix`](crates/rules/src/matrix.rs), over a trait of the
  four operations: the interpreter runs them on floats, the JIT feeds
  them a Cranelift builder, and the GPU prints the inverse through them
  as a WGSL function (the rest is WGSL's own), so the CPU and the JIT
  agree bit for bit and the GPU to rounding. Proof: the
  [feature matrix](crates/matrix/src/lib.rs),
  [diagnostics](crates/compiler/tests/compile.rs), and the
  [model example](examples/model/kernels/model.orla)'s orbit, one matrix
  applied to every vertex and normal.

- Integer vectors `IVec2/3` are value types of the ABI (`Type::lanes`
  gives a vector of either kind its count and lane type); every integer
  instruction of the interpreter takes a width and strides, the JIT zips
  words, and WGSL has them natively (bit operators and shifts widen the
  `Int` side, since WGSL wants both operands of one type). An integer
  address to `at`, `put` or a scatter is the float one converted by the
  compiler, exact for any coordinate a Field can hold, so the backends'
  Field paths did not change; `icoord`, `isize` and `idomain` are the
  float builtins converted the same way. Proof: the
  [feature matrix](crates/matrix/src/lib.rs),
  [diagnostics](crates/compiler/tests/compile.rs), and the
  [snake example](examples/snake/kernels/snake.orla), whose cells are
  integer vectors end to end.

- Vectors join and split in the compiler alone: `vec4(v3, w)` and the
  other constructors from shorter vectors flatten their arguments to
  components, and a swizzle `v.zyx` is extracts and a construct, so no
  backend has a swizzle path. A function may be named after a math
  builtin and shadows it in its document; the builtins with forms of
  their own are reserved (`RESERVED_BUILTINS`,
  [lowering](crates/compiler/src/lower/mod.rs)).

- Control flow is structured and expression-shaped: `if c { a } else { b }`
  and `loop N with x = init until done { next }` lower to [`Op::If` and
  `Op::Loop`](crates/ir/src/lib.rs) with pre-order value ids and block
  scoping proved by [the validator](crates/ir/src/validate.rs). Only the taken
  branch executes and loop bounds are compile-time constants
  ([execution tests](crates/cpu/tests/execute.rs),
  [scoping diagnostics](crates/compiler/tests/compile.rs)). The
  [ray-march example](examples/raymarch/README.md) is the working proof.

- `&&` and `||` ask the other side only where the first settles
  nothing. They lower to an `if`, which every backend runs one arm of,
  so `n != 0 && x / n > 1` does not divide when `n` is zero (it did,
  and the interpreter and the JIT called it a failure while the GPU
  computed something). The optimizer writes the cheap ones back as one
  operation before it hoists (`folded`: the other side at most four
  instructions, pure, and unable to fail), so nothing branches for a
  comparison and a loop can still lift one out; measured by alternating
  the two binaries over the replay suite, the change costs nothing.

- An `if` with no `else`, among a block's statements, guards what
  follows: `if x < 0 { return -1; }` is the `if` whose other arm is the
  rest of the block, desugared in the parser (`block_body`), so a
  Kernel reads like the check it is. Nothing in the IR knows.

- A `match` arm takes a range of values, `0..8` stopping before its
  end and `0..=7` taking it, and a `let` takes the fields of one value,
  `let { x, y: top } = window();`. Both are the parser's work and
  nothing else's: a range is the values it holds, written out, so the
  arms stay a table and a value in two arms is still refused by name;
  a pattern binds the value once (under a name no one can write, unless
  it is already a name) and reads a field for each binding, so a trace
  shows what one would have written by hand. A range holds at most 256
  values, which is a byte of opcodes; beyond that the message says an
  `if` on the bounds is the form.

- `match` on an `Int` is an op of its own (`Op::Match`: the scrutinee,
  arms of values and blocks, an `otherwise` block), typed by the
  validator (no value in two arms, one result type), run by the
  interpreter as a scan of the arms, emitted by the GPU as a WGSL
  `switch` with literal selectors and a `default`, and by the JIT as a
  `br_table` over a dense range (values spanning at most four times
  their count plus sixteen) or a chain of compares otherwise; the
  CHIP-8 and Thumb dispatches are written with it and their replays
  hold on the three backends. A nested loop reads an enclosing loop's
  carried value, and a scan's step its own inside a loop, because the
  compiler reads those values through `Carried` just before the inner
  loop begins and binds the names to the results, so no backend changed
  ([lowering](crates/compiler/src/lower/mod.rs),
  [interpreter](crates/cpu/src/eval/compile.rs), [WGSL](crates/gpu/src/wgsl.rs),
  [JIT](crates/jit/src/codegen.rs)).

- A function may name type parameters, `fn twice<T>(v: T) -> T`. Each
  is bound by the type of the argument written for it and must be the
  same everywhere it appears; one instance is compiled per set of types
  a call gives it, through the same machinery a Field parameter already
  used (`instantiate`, keyed now by the bound ports and the bound
  types). The body is checked at the instance, so an error there says
  which one was being compiled: "cannot apply `+` to Bool and Bool (in
  `twice` with `T` as Bool)". Nothing below the compiler sees a type
  parameter; the IR holds `twice<Float>` and `twice<Vec3>`, two
  functions.

- A `break` among a loop body's statements stops the loop after that
  iteration and gives that value: `if found { break i; }`. It is the
  compiler's work alone. A body that breaks carries two things, the
  value and whether it stopped, and the loop's exit test reads the
  second, so the IR, the interpreter, the JIT and the GPU learn
  nothing. Every path of the body gives that pair (`result_with_break`
  walks the `if`s and `match`es of the result position, so each path is
  still lowered once and only the one taken runs), and a `break`
  belongs to the innermost loop, since a nested `loop` is not walked
  into. What it costs is one carried word: at 960x540 the showcase's
  ray march measured about one percent slower written with a `break`
  than with the `until` it already had, so the hot loops there keep
  their form. If that ever matters, the IR's loop could take the flag
  as a value the body computes rather than as part of what it carries.

- A loop's count may be any `Int` expression
  ([`Op::Loop::bound`](crates/ir/src/lib.rs)), read once before the loop
  and clamped to `0..=MAX_LOOP_COUNT` — `smin`/`smax` on the JIT, a
  `clamp` in the WGSL bound, a clamped counter on the interpreter — so a
  network's width or a solve's depth is a parameter, not a literal.

- `[each N { body }]` builds an array element by element, unrolled, with
  `index()` its position; a loop inside the body keeps its own.

- Vector math and the clock: `Vec3`, `dot`, `cross`, `normalize`, `pow`,
  `step`, `smoothstep`, `reflect`, and `time()` are typed in
  [the IR](crates/ir/src/lib.rs) and exercised by
  [execution tests](crates/cpu/tests/execute.rs). The componentwise
  builtins (`floor`, `ceil`, `round`, `sign`, `fract`, `abs`, `sqrt`,
  `sin`, `cos`, `tan`, `asin`, `acos`, `atan`, `exp`, `log`) take a
  `Float` or any vector; `min`, `max`, `clamp`, `mix`, `pow`, `step`, and
  `smoothstep` broadcast a `Float` beside a vector; both backends agree
  ([parity](crates/gpu/tests/parity.rs)). `orla check` compiles the
  Kernels of every document a scene uses, not only the scene's own. `time()` needs `time.read`;
  only nodes that declare it re-execute when time advances
  ([clock caching test](crates/graph/tests/composition.rs)).

- The builtins were completed where a shader language or a machine
  expects them: `trunc`, `exp2`, `log2`, `sinh`, `cosh`, `tanh`, `fma`
  and the integer primitives `popcount`, `clz`, `ctz`, `rotl`, `rotr`,
  `ult`, `udiv`, `urem`, `mulhi`, `umulhi` are intrinsics on every
  backend (the GPU rotates and takes the high product half through
  emitted helpers, having no 64-bit ints; a zero divisor fails as `/`
  does), while `saturate`, `distance`, `radians`, `degrees`,
  `inversesqrt`, `mod`, `isnan`, `isinf` and `refract` are lowered by
  the compiler to the operations they are spellings of, so the backends
  have them by construction and agree by the same ops. Two GPU facts
  the matrix measured on the way: this GPU's native `u32 / u32` is
  wrong for dividends with the top bit set (0xFFFFFFF0 / 1 came out
  0xFFFFFFFE) while its signed division is exact, so `udiv` and `urem`
  are emitted through the signed one on halved dividends; and a
  driver's fast math may fold `inf * 0` to a number where the CPU has
  NaN, so a NaN made by arithmetic on the GPU is not promised; one that arrives in a Field is seen alike
  ([composed](crates/compiler/src/lower/mod.rs), [matrix](crates/matrix/src/lib.rs)).

- `orla help` and the protocol's `help` list what a Kernel may call:
  the math intrinsics' signatures are found by asking the IR's typing
  rule about every combination of the plain types, their sentences live
  beside the enum, and the Field, memory, context and construction
  builtins are described next to the lowering that dispatches them
  ([help](crates/compiler/src/help.rs)), so the listing is what
  compiles.

- A call to a function that is not above the caller now says why: a
  call to itself says a Kernel has no stack and a `loop` is the form,
  and a call to one defined below says a function calls only what is
  above it. Both were "is not a builtin, intrinsic, or function defined
  above" before, which reads as a typo.

- Undeclared capabilities are compile-time diagnostics with positions
  ([compile tests](crates/compiler/tests/compile.rs)); the IR validator remains
  the backstop.

- Diagnostics that cross files: a `use` that comes back to the document
  it started from is reported as the cycle it is (`declarations_within`
  carries the origin and the open path, so `a.orla` and `b.orla` use
  each other reads that way rather than as a name declared twice), and a
  name a Kernel cannot find is looked for beside the document, beside
  the scene and beside every library the session knows, so the
  diagnostic ends with the file that declares it and the `use` that
  would bring it in.

### Context: input, window, sound

- Input is context, like time: `pointer()`, `pointer_down()`, `button(n)`,
  `wheel()`, `stick(n)`, and `key(code)` behind `input.read`; only nodes
  that declare it re-execute when input changes ([ABI](crates/abi/src/lib.rs),
  [parity](crates/gpu/tests/parity.rs)).

- A pad is context too, not a device a scene opens: `stick(0)` is the left
  stick, `stick(1)` the right, `stick(2)` the two triggers (`x` left, `y`
  right), each pair in the context beside the pointer, and the pad's
  buttons are `button(8)` upwards ([`orla_abi::pad`](crates/abi/src/lib.rs)
  names them). The Player fills them from gilrs with a dead middle taken
  out, and still presses the arrow keys from the left stick, so a scene
  written before a pad existed plays on one
  ([gamepad](crates/player/src/gamepad.rs)). A replay drives them
  (`stick left 0.5 -0.25`, `button south down`) and the protocol takes
  `"sticks": [[lx, ly], [rx, ry], [lt, rt]]`, so what a pad does is
  testable without one ([replay](crates/cli/src/replay.rs),
  [protocol](crates/cli/src/protocol.rs), [feature matrix](crates/matrix/src/lib.rs)
  `context`).

- The window is context too, a device fact like the sound card's rate:
  `Context` carries its size in physical pixels and its scale factor, a
  node's `domain` may take them (`domain = window();`) and so may a
  Kernel under `input.read` (`window()`, `scale()`), and since the plan
  is drawn every evaluation and the extent is part of a node's cache
  key, a resize re-executes the nodes it sizes and nothing else
  ([domain](crates/graph/src/domain.rs), [evaluator](crates/graph/src/evaluate.rs),
  [graph test](crates/graph/tests/composition.rs)). The Player hands its
  window over each frame, `run` takes `--window`, and a replay declares
  one ([replay](crates/cli/src/replay.rs)); the voxel editor and the
  showcase fill the window at its resolution with their bars laid out in
  units of `scale()`, and their replays run at a fraction of both.

- A gamepad is another keyboard ([gamepad](crates/player/src/gamepad.rs)):
  the Player maps the D-pad, the left stick past a threshold, and the
  face and Start buttons to the key codes a keyboard would produce, so
  input stays one context and scenes need not know which device pressed
  the key. Analog values are not context yet.

- The sound card is a device adapter in the Player, not a runtime concept
  ([audio](crates/player/src/audio.rs)): it claims the first graph output
  that is rows of one or two floats, takes each evaluation's block into a
  lock-free ring, and feeds the device from the ring in its real-time
  callback, which never waits on the Graph or the GPU and never
  allocates; a dry ring plays silence and counts an underrun. The device's
  clock is context (`Context.rate`, read by `rate()` under `time.read`),
  and while a device plays it paces evaluation: the Player computes the
  next block when the ring can take it. Materializing
  the block is the observation boundary, on the frame thread. The
  [audio example](examples/audio/README.md) carries the oscillator's
  phase across blocks through `<~` state, and its replay pins the
  oscilloscope, block continuity included.

### Files, assets and tools

- Assets are inputs with files: `input atlas: Image = "atlas.ppm";` names
  a file the session reads into the Host and re-reads when its stamp
  changes, so a picture edited on disk changes the running scene; `.png`
  and `.ppm` are grids of pixels (or `Grid<Float>` luminance when the
  input asks), `.pgm` a `Grid<Float>`, `.wav` rows of one or two floats
  with the rate reported, `.field` any Field in the Host's own format
  ([file](crates/host/src/file.rs)). PNG and WAV are decoded and encoded
  with no dependency ([inflate](crates/cli/src/inflate.rs),
  [png](crates/cli/src/png.rs), [wav](crates/cli/src/wav.rs)), checked
  against reference vectors; `run --save` and the protocol's `read`
  produce them ([session](crates/cli/src/session.rs),
  [tests](crates/cli/tests/session.rs)). Text is an input too: `input
  title: Rows<Int> = text "ORLA";` is the characters' codes as rows the
  session writes from the document and rewrites when it changes
  ([document](crates/graph/src/document.rs), [session test](crates/cli/tests/session.rs)),
  which the [text example](examples/text/README.md) lays out and draws.

- Text needs no primitive: a label is `Rows<Glyph>` (code, origin,
  scale), a font is a `Grid<Float>` atlas the Host reads from a file, and
  `text.draw@1` paints one over a picture through a shared function any
  Kernel may call; `text.layout@1` and `text.digits@1` make the rows
  ([library](packages/text.orlapkg/kernels/text.orla), [font](packages/text.orlapkg/kernels/font.orla),
  [text example](examples/text/README.md)). The atlas is baked by a
  Kernel through `run --save` and both backends draw it bit for bit
  ([replay](examples/text/tests/text.replay)).

- A file as its bytes: `input level: Rows<Int> = bytes "level.bin";`
  reads any file as rows of one integer an element, 0 to 255, watched
  and re-read like any other asset. What the bytes mean is the Kernel's
  business, so a format of one's own needs no reader in Rust: the
  `level` example walks a header, a tile map and an entity list, and a
  count read from the file is a loop's count, which the compiler still
  bounds. Formats other software writes keep their readers in
  `orla-import`, where the dependencies are.

- Frames in (phase 5, video's input half): `input clip: Image = frames
  "dir/" at N;` lists a directory's frame files in name order on sync
  (`Sequence` in the session) and, at each evaluation, brings in frame
  `floor(time * N)` wrapping, as the external; a frame that cannot be
  read is reported by the next sync and the input keeps its frame.

- A score in and out (phase 5, audio): `orla-import` reads a standard
  MIDI file into rows of `{start, duration, pitch, velocity, channel}`
  with the ticks and the tempo map already turned into seconds, and
  `orla-export` writes those rows back as a file a music editor opens.
  Both are written from the format's own definition and take no
  dependency; the reader keeps a note's second at its strike, since the
  tempo may change again while it sounds. A score read and written
  again is the same rows.

- Models and compressed sounds come in through a tool: `orla-import`
  ([crate](crates/import)) sits in the interface layer beside the CLI and
  the Player, the only layer that takes dependencies. `import` reads
  glTF/GLB (external, data-URI or GLB buffers; every accessor type; node
  transforms baked; normals computed when absent) and writes vertex and
  triangle rows as `.field` and the base colour texture as PNG
  ([gltf](crates/import/src/gltf.rs), [model example](examples/model/README.md)),
  and reads MP3, OGG Vorbis, FLAC, AAC and more through symphonia and
  writes a WAV at the file's rate ([audio](crates/import/src/audio.rs),
  checked against tones made by ffmpeg). The runtime sees none of these
  formats. `orla-export` ([crate](crates/export)) is the way out: vertex
  and triangle rows and a PNG as a GLB, rows of floats as a WAV, grids as
  PNG; the round-trip test brings the model example's crate back through
  the importer bit for bit ([gltf](crates/export/src/gltf.rs),
  [tests](crates/export/tests/round_trip.rs)). Lossy audio is not
  encoded here: WAV is the lossless way out and ffmpeg compresses it.

- Render (phase 5, video): `orla render` steps a scene at a fixed rate
  for a count of frames, the clock being the frame count over the rate,
  and writes each picture as a PNG in a directory and the sound output as
  one WAV, block after block; `async` nodes are settled after every
  frame; `--save` writes a node's output after the last frame, how a
  state reached over many frames (a network's weights) is kept for a
  scene to read back. A video is ffmpeg's job over the directory; no
  codec is a dependency.

### Interfaces: protocol, C ABI, replay

- The creation protocol is an interface over the session, not a new
  authority: `prepare` never publishes, `publish` checks an expected
  generation, `compose` rewrites the scene document so the file stays the
  source of truth, and diagnostics carry line and column
  ([protocol](crates/cli/src/protocol.rs)).

- The protocol has a socket transport: `--listen <host:port>` on
  `orla serve` and the Player feeds every connection's lines to the one
  session with per-connection replies, a socket's `quit` ending only its
  connection ([protocol](crates/cli/src/protocol.rs),
  [test](crates/cli/tests/protocol.rs)). A connection that opens with an
  HTTP upgrade speaks WebSocket frames instead, the handshake and framing
  of RFC 6455 written out with no dependency ([websocket](crates/cli/src/websocket.rs),
  [client test](crates/cli/tests/protocol.rs)). `bridges/blender` is the
  first bridge: a Blender add-on in Blender's own Python that writes
  meshes as the Host's files and talks to a running window over the
  socket; [`bridges/web`](bridges/web/README.md) is the second, a page in
  JavaScript that shows the frames, sends the canvas as the window and
  the input, and publishes Kernels, checked headless by
  [its runner](bridges/web/test.mjs);
  [`bridges/godot`](bridges/godot/README.md) is the third, a Godot 4
  project in GDScript: a blocking client over `StreamPeerTCP`, a Control
  that is the window and draws each frame as a texture, an `ArrayMesh`
  from two nodes' rows, the graph's params as controls, checked by
  [its test](bridges/godot/test.gd), which spawns the sessions itself
  through Godot's pipes; bridges are clients outside the workspace.

- The same session is a library in another program's process:
  `orla-ffi` ([crate](crates/ffi), [header](crates/ffi/include/orla.h))
  builds a `cdylib` and a `staticlib` with ten `extern "C"` functions
  over `Session`: open a scene on a backend, evaluate under a
  `#[repr(C)]` input (time, pointer, buttons as bits, keys as bits,
  window and scale), read an output's or a pinned node's Field as a
  pointer into the Host's words with its extent, stride, shape and
  schema (a picture is RGBA32F in place, no PPM and no copy), the last
  report as JSON, and one protocol line in and one out for everything
  else. Strings belong to the session until the next call, a Field's
  words until the next evaluate, request or close; one thread at a time
  per session; every entry catches a panic and reports it as an error;
  a NULL session is inert. The header is written by hand, no generator.
  The tests drive the functions from Rust as C would, and
  [`bridges/c/hello.c`](bridges/c/hello.c) links the library from C
  ([ffi](crates/ffi/src/lib.rs), [tests](crates/ffi/tests/abi.rs)).

- Application tests are data: `.replay` files script frames, pointer, keys,
  reset, and pixel expectations; `orla replay` runs them and the interface
  tests run every `examples/*/tests/*.replay`
  ([replay](crates/cli/src/replay.rs)).

- Automation past one frame (phase 5): the protocol's `advance` runs a
  count of frames in one request, stopping when a watched value says
  to and saving node outputs when it stops. Reading that value is the
  session's `element_field`, which a replay's `expect` uses too, over
  the same `node.port[index].field` spelling (`parse_target`); the run
  demands only what it watches, so a training loop does not draw the
  picture. An agent's loop becomes one request per checkpoint rather
  than one per frame — the [rubik](examples/rubik/README.md) network
  trains to a loss it names, in one request.

### Runtime and backends

- Backends implement [`Backend`](crates/runtime/src/lib.rs): `prepare` does
  the expensive work (the GPU backend compiles WGSL and builds the pipeline
  there), so `publish` is a pointer swap. GPU and CPU results agree within
  float tolerance (exactly on the examples, whose hashes avoid `sin` of
  large arguments), and integer arithmetic failures are reported only by
  the CPU backend.

- The JIT is the default backend everywhere but the Player, which takes
  the GPU, then the JIT, then the interpreter. `--backend cpu` and `trace`
  ask for the reference rather than falling into it.

- Preparation leaves the frame thread: [`Runtime::prepare_later`](crates/runtime/src/lib.rs)
  hands a validated module to one worker and returns a ticket; `poll`
  collects the candidate, and publishing stays a pointer swap on the owning
  thread. Without a worker the same calls prepare inline, which is what
  `run`, `replay`, and `check` use for determinism; `serve` and the Player
  enable the worker after the first load. The session republishes a running
  Kernel through a ticket (`preparing`, then `published`) and keeps the old
  version until then; the evaluator runs a group's nodes separately until
  its fused version arrives, which changes no pixel
  ([runtime test](crates/runtime/tests/hot_swap.rs),
  [graph test](crates/graph/tests/composition.rs),
  [session test](crates/cli/tests/session.rs)). wgpu's error scopes are
  thread-local only with its `std` feature, which the GPU backend therefore
  enables ([parity test](crates/gpu/tests/parity.rs)).

- The CPU backend compiles a module to word-addressed code once per
  execution and runs it once per element: every value has a fixed word
  offset in the ABI's own layout, so arguments, struct fields, vector
  components and a loop's carried value are aliases that cost nothing,
  Field reads copy words out of slices the scope lent once, and nothing
  allocates per element ([interpreter](crates/cpu/src/eval/compile.rs)). Rows run
  in parallel and are written sequentially; results and error attribution
  match a single-threaded run ([parallel rows test](crates/cpu/tests/execute.rs)),
  and the GPU parity tests and every example replay hold the two backends
  to the same pixels.

- The interpreter's word code built once, not once per execution
  ([`Code`](crates/cpu/src/lib.rs), [`Program::compile`](crates/cpu/src/eval/compile.rs),
  [`CpuCode`](crates/runtime/src/lib.rs)): `prepare` used to build nothing,
  so every `execute` laid out every value's words again — about seventy
  nanoseconds a statement, paid per frame per node however small the
  domain. The compiled code is what the backend's `Prepared` holds now,
  and `execute_with` binds an invocation to it. A thousand-statement
  Kernel over one element went 80 to 10 microseconds a frame; a
  hundred-and-twenty-statement one over sixteen elements 90 to 60, the
  rest being real arithmetic. `CpuBackend::execute` still compiles on the
  spot for a direct caller, and `trace` compiles its own without the
  optimizer. The JIT was checked and already compiles in `prepare`.

- The JIT backend ([crate](crates/jit)) is the interface layer's third
  backend and the one crate that takes `unsafe`: it compiles a validated
  module's functions to native code through Cranelift when the Kernel is
  prepared (every value its words in the ABI's layout, `if` and `loop`
  as blocks with block parameters, a function's result through a
  caller-provided slot, integer arithmetic checked as the interpreter
  checks it with the first failure recorded in a context struct, Field
  reads and the transcendental functions through Rust helpers by symbol
  so they round the same), and runs a map's entry once per element over
  the CPU's threads. A scan compiles too: its `init` is a function
  without arguments, its entry takes the step's index and a pointer to
  the carried words (`Carried` and `Index` outside every loop read them),
  the memories get working copies the views point at so a step's reads
  see its writes, `put` goes through a helper that resolves the address
  as the interpreter does and drops a write outside the memory, and the
  steps run in order on one thread; the copies become the outputs after
  the first. On the GPU the fold pass carries the map pass's bindings
  (the context, the params, every input) before its own level buffers,
  so a `merge` that reads a Field, a param or the time compiles there as
  it runs on the CPU and the JIT ([fold](crates/gpu/src/wgsl.rs),
  [parity test](crates/gpu/tests/parity.rs)). A reduction compiles its `merge` as a function taking its
  two values through pointers, so the host folds the map's words in the
  interpreter's fixed tree by calling it whatever the words; a scatter
  runs its entry over the sources into pairs and applies them in element
  order with the CPU crate's own combine rule. The native operations,
  the rasterizer, are the CPU's code. The parity tests hold the map
  vocabulary, a scan with memories, a reduction and scatters bit for bit
  and every example replay passes on `jit`; the raymarch, voxel and showcase scenes run twenty to thirty-five
  times faster than the interpreter and the CHIP-8 steps eight to nine
  times faster ([lib](crates/jit/src/lib.rs),
  [codegen](crates/jit/src/codegen.rs), [tests](crates/jit/tests/parity.rs)).

- A runtime may run its scans on a second backend (`Runtime::with_backends`):
  a module is prepared on and executes on the backend its mode
  chooses, the preparation worker included; a scan's inputs are brought
  into Host memory before it runs and its outputs are Host Fields the
  main backend uploads when it reads them. A `gpu` session pairs the GPU
  with the JIT this way, so ordered work is native code and the picture
  stays on the device; the mixed runtime is tested with two recording
  backends ([runtime](crates/runtime/src/lib.rs),
  [test](crates/runtime/tests/mixed.rs), [session](crates/cli/src/session.rs)).

- The GPU divides integers through a dividend the shader compiler
  cannot narrow: a one read from the context (`wgsl::OPAQUE_ONE`) is
  multiplied in by `whole_i32` and its vector and unsigned kin before
  every `/` and `%`. Metal took the top sixteen bits of a word
  (`x >>> 16`, or `(x >> 16) & 0xFFFF`) as a sixteen-bit value and
  divided it by a constant wrongly (`0 % 6` came out 65530); the
  [feature matrix](crates/matrix/src/lib.rs)'s `narrow division` row
  fails without the guard. In a Graph, `size(port)` of another node's
  memory port is the memory's extent, what it committed last or its
  initial value's size, not the node's domain
  ([evaluator](crates/graph/src/evaluate.rs)).

- Async nodes (the plan's "async Kernels"): `async = true;` on a node.
  `Runtime::execute_later` copies the invocation's inputs into a Host of
  the thread's own and runs the backend's execute there (a scan on the
  ordered backend, as on the frame); `poll_run`/`wait_run` adopt what
  landed as Fields of the runtime's Host, so no Host is shared between
  threads and nothing in the backends changed. The evaluator keeps one
  run per node, keyed like a cache entry: a run starts when the key no
  longer matches and none is going, the frame carries the cached output
  meanwhile, commit skips the node's state while it runs, and the landed
  outputs go into the cache under the key they started with, so the next
  evaluation reuses them, re-runs the readers, and commits. A first run
  is on the frame; such a node is never fused. Not a parallel scheduler:
  measured, a frame is the sum of its nodes on all three backends and a
  map already uses every core, so node-level threads would gain nothing;
  what was missing was a node that need not finish inside the frame.

- A Kernel can be looked into: every function keeps its `let` bindings
  as a side table of names to values (`Function.bindings`, no part of
  the program's meaning, renumbered through fusion), the interpreter
  keeps every value's frame location, and `orla_cpu::trace` runs one
  element (a scan's step after the steps before it, on a frame cleared
  first) and reads the bindings back, marking which ran by walking the
  entry with the `if` conditions' values (a loop's body counts as run
  when the loop did). The session evaluates first with the node's
  sources pinned, builds the node's invocation as an evaluation would
  and traces on the interpreter without committing; `orla trace` prints
  it, the protocol's `trace` answers it as JSON, and the evaluation
  report times every executed node (`run --profile`)
  ([trace](crates/cpu/src/lib.rs), [session](crates/cli/src/session.rs),
  [protocol](crates/cli/src/protocol.rs)).

### The optimizer

- The IR optimizer (the plan's section 41) runs between lowering and
  validation, four exact rewrites: a part taken straight from its
  construction is that part and a vector rebuilt from all its parts is
  the vector (what swizzles and joins leave behind); an operation on
  literals is its literal where the backends' arithmetic gives one and
  no failure (an overflowing or dividing-by-zero integer expression
  stays for the run to report); a pure operation repeated on the same
  operands in one block or an enclosing one is shared (a read of a
  memory the steps write, a call to a writer, and a loop's own carried
  value are not); a value nothing reads is dropped unless it is a
  write, holds one, or is a name a trace shows. Values are renumbered
  in pre-order. Held to the interpreter word for word over the feature
  matrix ([test](crates/cpu/tests/optimizer.rs)), which is what makes
  the rewrites safe to trust: roughly a quarter of the instructions go,
  and the JIT gains where they were Field reads, since Cranelift had
  already done the rest. `orla check --ir` shows the optimized IR. Since then: short functions
  written out where they are called, work lifted out of a loop that
  cannot fail, a short-circuit written back as one operation, and a
  loop run here. That last one (`Simplify::run_loop`) takes a loop
  whose count is a literal, whose initial value is a literal and whose
  body only computes, runs it while the Kernel compiles, and leaves the
  literal it gives. It walks the body's `if`s, `match`es and inner
  loops, reads `Op::Carried` and `Op::Index` as the run has them, and
  stops at anything else: a Field read, a memory write, a call, or an
  operation that would fail, since a failure belongs to the element
  that has it. Twenty thousand operations is the budget for a function,
  which a table built in a loop spends a hundred of. What it buys is
  that the clear way to write a table — a loop over constants, as
  [collapse](examples/collapse/README.md) writes it — is also the cheap
  one.

- Short functions written out where they are called (the optimizer's
  fifth rewrite): a call costs the interpreter a frame and the JIT a
  real call, and a function of at most 96 instructions is cheaper
  spelled out. Functions are taken in order, so a call is always to one
  already optimized and depth costs nothing; the callee's values take
  fresh ids, its `Arg(i)` reads become the call's operands, and what it
  returns takes the call's place. It is worth most where a small
  function is called per element; the JIT and the GPU are unchanged. A
  trace compiles the Kernel again without
  the optimizer (`compile_kernel_unoptimized` over the parsed Kernel the
  session keeps), so it still walks the source's own calls and `let`s.

- Work a loop repeats without changing, done once before it (the
  optimizer's sixth rewrite): an instruction in a loop's body whose
  operands all come from outside gives the same value every iteration.
  Only operations that cannot fail move, so a run that fails still
  reports the failure it did, of the element it did; integer arithmetic
  can overflow, so it stays. Innermost loops are lifted first, so what
  an inner loop sheds can leave the outer one too. It is worth most to
  the interpreter; the JIT and the GPU are unchanged by it, since their
  code generators already hoist what they can.

- Two float operations as one interpreter instruction (`FBin2`): a
  value that a single following operation reads never reaches the frame
  and one dispatch does the work of two. Only the instruction just
  emitted folds into the next, since anything in between may write the
  slots it read. It is worth a few per cent: the interpreter's cost is
  its frame traffic and its loop bookkeeping more than its dispatch
  count, which is why the rest of the gap to the JIT is structural.

- A field or an element taken out of a whole element of an input is a
  read of those words alone ([`Op::Part`](crates/ir/src/lib.rs),
  [`narrowed`](crates/ir/src/optimize.rs)): the instruction holds the
  way from the element to the part, field by field and index by index,
  and each backend reads only the words the path names — the
  interpreter one copy, the JIT a load at a worked-out offset (from the
  place the run already carries, where the element is this one's own),
  the GPU an indexed read of the storage buffer. The element is the one
  an address names or, with no address, this element's own, which is
  what `sample` reads; a fused producer leaves nothing to read, so
  fusion writes such a part out as the extracts and subscripts it
  stands for ([`fuse`](crates/ir/src/fuse.rs)). Without it a Kernel whose
  input element is a record of arrays copies every word of an element
  into every element of its own domain to reach three of them, which is
  the memory system and not the work. The whole read stays where something wants the whole (a value chosen by
  an `if`, say); a name is not such a want, so the `let` goes with the
  read it named, and a trace of a Kernel a document owns walks the
  unoptimized source where every `let` still is. Memories are left
  alone: a `put` between the read and the part would be seen by the
  part and not by the copy. A function taking a record or an array of
  32 words or more is written out where it is called however long it
  is (up to 1024 instructions), since the argument is copied at every
  element and, written out, its reads narrow too.

### What a device costs

- The buffer pile serves a buffer that is big enough, not one of
  exactly the size ([`Spare::take`](crates/gpu/src/lib.rs)): nothing
  reads past the element count the uniform gives — no emitted shader
  asks a buffer how long it is — so the tail of a bigger one is never
  touched, and the smallest that fits is taken, never one more than
  twice the size. It is what makes the pile work for a scene whose
  Fields change size, which is every scene that sizes a node by what a
  Kernel decided: `culled` keeps a different number of chunks every
  frame, so an exact match never hit and every geometry buffer came
  from the driver. Scenes whose Fields are a fixed size are unchanged.

- What a node costs beyond its work, measured and left alone: on the
  GPU most of it is the device's own launch of a pass, not the
  backend's recording, which hides behind the device working on the
  node before. Fusion is the only thing that takes a pass away, and it
  already does. Batching a frame into one submission is unsound, not merely unprofitable — the context and
  parameter uniforms are one pair of buffers rewritten per node with
  `queue.write_buffer`, which orders against submissions and not
  against recording, so a batch's nodes all read the last one's
  uniforms.

- A reduction folds sixteen at a time, in the same tree
  ([`orla_rules::reduce`](crates/rules/src/reduce.rs),
  [`emit_reduce`](crates/gpu/src/wgsl.rs)): the tree is pairwise with an
  odd last element carried up, and a run of sixteen folded by that rule
  reaches the same expressions as four of its levels walked one at a
  time, ragged ends included — which is what lets a device do four
  levels in one dispatch and change no value: a level is a dispatch with
  a bind group and a uniform of its own, and folding sixteen at a time
  takes twelve of them down to three. The interpreter and the JIT still
  walk one level at a time — sixteen merges in a row are a dependency
  chain, and folding a run left to right rather than as a tree is slower
  on a machine that can run the pairs at once. The level buffer and a
  scatter's target come from the buffer pile, as a full-screen pass
  does.

- A profile's numbers are the node's, and they add up
  ([`Runtime::execute`](crates/runtime/src/lib.rs),
  [`measurement_floor`](crates/runtime/src/lib.rs),
  [`node_times`](crates/cli/src/bench.rs)): a native operation was never
  measured — the runtime set `last_measured` only in the Kernel branch,
  so a rasterize was printed with whatever its neighbour had left
  standing, and it never waited for its own submission, so the work it
  had queued was billed to the node after it — which read as the
  cheapest node in the frame being its most expensive. The number is
  cleared before every execution and
  asked for after a native one too; the rasterizer waits under a
  measured run and carries timestamps across its two passes. The
  device's own clock is asked once and remembered — Metal through wgpu
  resolves timestamps to zeros, and asking again per node cost a mapped
  buffer and a polled queue every time — and what measuring one node
  costs before it does anything (`Backend::measurement_floor`, one
  empty submission polled to completion) comes off every line. A profile says which kind of number it is printing and prints
  the sum against the frame, since a node measured alone does not
  overlap the ones around it.

- A measured profile: `run --profile` puts the GPU backend in a
  measured mode, where an execution waits for its submission and asks
  the device for the time between a node's first pass and its last
  (timestamp queries at the passes' edges, since writing one inside an
  encoder is a feature Metal does not offer). `Backend::last_device_time`
  carries it to the runtime and into the report's timings, so the
  numbers are the device's own clock where the driver honours the
  queries, and the work plus one round trip where it resolves zeros,
  which is what Metal through wgpu does today. Since that round trip is
  counted per node, a profile says which node is heavy and not what a
  frame costs; `bench` answers that. The Player's F10 line asks the
  plainer question every frame while it is up: it waits for the device
  once evaluating has given it everything
  ([`Runtime::settle_device`](crates/runtime/src/lib.rs)) and shows what
  that took as `device` (not `gpu`: the backend's own name is already
  that, further along the same line), so the device's work is a number
  of its own rather than a share of `present`, where the display's
  refresh is. A backend
  with no device of its own ([`Backend::has_device`](crates/runtime/src/lib.rs))
  has no such field. Asking costs the overlap between one frame's device
  work and the next frame's own, which is why nothing but the line
  asks.

- An iterative solve costs its iteration count times the floor every
  node pays, and most of a Jacobi step is that it is a step. A `times`
  that stopped when the residual was small would not help: a solver's
  step count is the budget its author chose, not the point where it
  converged, and fewer, bigger iterations are an algorithm's business,
  not the substrate's.

- Whether a handle is still held is a count, not a walk
  ([`cached_handles`](crates/graph/src/evaluate.rs)): a scene of state
  asks once a binding a frame, and the answer walked every cached
  entry's outputs, so a graph of a thousand states was squared — 2.5
  microseconds a node at sixty-four and 8.0 at a thousand. Two counts
  of how many cached entries and committed ports hold each handle,
  kept by the four methods that touch the two maps so nothing else has
  to remember, and a `debug_assert` that they still agree with the
  walk, since a count that drifted would leak a Field or release one
  twice. `commit` also tested "has this port advanced" against a list;
  it is a set beside the list now.

- Finding a node by name is an index, not a walk
  ([`Graph::index`](crates/graph/src/lib.rs)): `Graph::node(id)` walks
  the node list, and the topological sort, the pruner and the planner
  each did it once a node, once a cached entry and once a binding — so
  a frame was squared in the node count. The index is built once and
  passed to the phase that needs it, the sort works out each node's
  dependencies once instead of on every step of its walk, and the
  planner keeps a map of where its groups are. It shows in a graph of a
  thousand nodes; no example has enough for a square to appear.

### Verification

- The feature matrix grew a second part: not what the language can
  write but what it promises. `not taken` puts a division by zero on
  every side that must not run (an `if`'s other arm, a `match`'s other
  arm, a short-circuit's other side, a loop stopped by its `until` and
  a loop of no iterations), so a run that finishes is the proof; break
  one and the interpreter says "integer division by zero" in all five
  modes. `edges` reads past every edge there is (a Field by `at` and
  `at_linear`, an array, a matrix's column) and holds the three
  backends to the same clamp. `signs` pins a quotient truncating toward
  zero and a remainder taking the dividend's sign, floats and the
  unsigned pair included. `float edges` pins an infinity where a
  positive is divided by zero, a not-a-number where zero is, and that
  nothing equals or orders against one; `sqrt` of a negative is left
  out, since a driver's fast math need not have NaN there, which the
  documents now say. A failure keeps naming its element through an
  inlined call and a loop, and work that can fail is still never lifted
  out of a loop that may not run.

- The backends are held to the interpreter across the whole cross
  product, not where an example happens to look: `orla-matrix` writes
  every feature of the language (arithmetic, bits, comparisons,
  vectors, the transcendental functions, rounding, conversions,
  structs, `if`, `match`, loops nested and with `until`, calls plain and
  with a Field parameter, Field reads of every kind, the context, the
  params, hex and negative literals) into every execution mode (a map
  over rows, a map over a grid, a reduction, a scan with a memory, a
  scatter) as complete Kernels with their inputs, and the JIT's and the
  GPU's matrix tests run all of them against the CPU, the JIT bit for
  bit, the GPU with ints and booleans exact and floats to rounding,
  reporting every case that differs at once. Its first run found two
  holes on the GPU: WGSL's float `%` rounds the quotient before
  truncating it (7.5 % 0.3 came out 0 against the CPU's 0.3 minus
  epsilon), so a Float remainder is now an emitted `fmod` built on a
  fused multiply-add with a one-divisor correction; and a scatter into
  rows named its address `i` in a pass that already had an `i`, so
  every scatter into rows was refused by the GPU. Both hold now
  ([matrix](crates/matrix/src/lib.rs), [WGSL](crates/gpu/src/wgsl.rs)).

Use [USAGE](docs/USAGE.md) for reproducible commands and [HANDOFF](HANDOFF.md) for
the next action. Git preserves completed work. Do not maintain feature
inventories, test counts, or completion percentages here.
