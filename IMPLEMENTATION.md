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
| Kernel hot-swaps without restart | [hot swap tests](crates/core/kernel/tests/swap.rs), [a scene's change swapped whole](crates/interface/session/tests/scene_change.rs), [session tests](crates/interface/entelechy/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/core/graph/tests/composition.rs), [example scene](examples/grayscale/scene.esn) |

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
| File writes | [writes](crates/interface/documents/src/writes.rs), [bundled saves](crates/interface/session/src/session/saves.rs); `else` fallback in the [evaluator](crates/core/graph/src/evaluate/mod.rs) | [session](crates/interface/session/tests/writes.rs), [bundle saves](crates/interface/session/tests/bundle.rs), [runner saves](crates/interface/player/tests/runner.rs), [lowering](crates/frontend/scene/src/lib.rs), [save replay](examples/data/tests/save.replay) |
| Laws a world writes | output `to files` into its law's document; [history origin](crates/interface/documents/src/history.rs), [kept store](crates/interface/documents/src/store.rs), replay `allow`; [template](packages/text.entpkg/graphs/template.esn) | [genesis tests](crates/interface/session/tests/genesis.rs), [genesis replay](examples/genesis/tests/genesis.replay) |
| Shared rules | [rules](crates/core/rules/src/lib.rs), [`Backend`](crates/core/rules/src/backend.rs), [reduce tree](crates/core/rules/src/reduce.rs), [rasterizer](crates/core/rules/src/raster.rs), [ray tracer](crates/core/rules/src/trace.rs) | every backend's parity test |
| Reference backend | [cpu](crates/backends/cpu/src/lib.rs), [word code](crates/backends/cpu/src/eval/) | [execution](crates/backends/cpu/tests/execute.rs), [optimizer](crates/backends/cpu/tests/optimizer.rs) |
| JIT backend | [jit](crates/backends/jit/src/lib.rs), [Cranelift emission](crates/backends/jit/src/codegen.rs), [WebAssembly emission](crates/backends/jit/src/wasm) | [parity](crates/backends/jit/tests/parity.rs), [matrix](crates/backends/jit/tests/matrix.rs), [as WebAssembly in node](crates/interface/entelechy/tests/wasm.rs) |
| GPU backend | [gpu](crates/backends/gpu/src/lib.rs), [WGSL](crates/backends/gpu/src/wgsl/mod.rs), [GLSL ES 3.00](crates/backends/gpu/src/glsl/mod.rs), [WebGL2](crates/backends/gpu/src/webgl.rs), [indirect](crates/backends/gpu/src/indirect.rs), [rasterizer](crates/backends/gpu/src/raster.rs), [ray tracer](crates/backends/gpu/src/trace.rs), [presenter](crates/backends/gpu/src/present.rs), [wait](crates/backends/gpu/src/wait.rs) | [parity](crates/backends/gpu/tests/parity.rs), [matrix](crates/backends/gpu/tests/matrix.rs) (skip without an adapter), [shader names](crates/backends/gpu/tests/names.rs), [WebGL2 in Chrome](bridges/web/test.html) |
| Conformance matrix | [matrix](crates/backends/conformance/src/lib.rs) | bit for bit [on the JIT](crates/backends/jit/tests/matrix.rs), ints exact and floats to rounding [on the GPU](crates/backends/gpu/tests/matrix.rs) |
| Kernel Runtime | [runtime](crates/core/kernel/src/lib.rs), [off-frame runs](crates/core/kernel/src/later.rs), [clock](crates/core/kernel/src/clock.rs) | [hot swap](crates/core/kernel/tests/swap.rs), [async](crates/interface/session/tests/later.rs), [mixed backends](crates/interface/session/tests/mixed.rs) |
| Scene documents | [scene](crates/frontend/scene/src/lib.rs), [domain](crates/frontend/scene/src/domain.rs) | lowering tests in place, [every form lowered](crates/frontend/scene/tests/surface.rs) |
| Graph Runtime | [graph](crates/core/graph/src/lib.rs), [evaluator](crates/core/graph/src/evaluate/mod.rs), [domain](crates/core/graph/src/domain.rs) | [composition tests](crates/core/graph/tests/composition.rs), [with the compiler](crates/interface/session/tests/composition.rs) |
| Documents | [documents](crates/interface/documents/src/lib.rs): [scene](crates/interface/documents/src/scene.rs), [kernels](crates/interface/documents/src/kernels.rs), [inputs](crates/interface/documents/src/inputs/mod.rs), [sources](crates/interface/documents/src/sources.rs), [catalog](crates/interface/documents/src/catalog.rs), [store](crates/interface/documents/src/store.rs); [formats](crates/interface/formats/src/lib.rs) | [layers](crates/interface/entelechy/tests/layers.rs) |
| Session | [session](crates/interface/session/src/session.rs), [bundle](crates/interface/session/src/bundle/mod.rs), [observe](crates/interface/session/src/observe.rs), [bench](crates/interface/session/src/bench.rs) | [session tests](crates/interface/entelechy/tests/session.rs), [bundle tests](crates/interface/session/tests/bundle.rs) |
| Creation protocol | [protocol](crates/interface/protocol/src/lib.rs): [inspect](crates/interface/protocol/src/inspect.rs) (read-only `status` with `last_frame` facts; `catalog` package/Kernel/preset search), [create](crates/interface/protocol/src/create.rs), [trial](crates/interface/protocol/src/trial.rs) (in-memory `Kept` shadow verification `trial` with compile/runtime/finite/budget/download gates, auto-trace, and atomic commit), [run](crates/interface/protocol/src/run.rs) (`evaluate` with multi-tick headless `steps` / `until`), [fields](crates/interface/protocol/src/fields.rs) (`read` tensor `summary` with `rms`, `min_at`, `max_at`, `nan`, `inf`, `first_nonfinite`), [wire](crates/interface/protocol/src/wire.rs), [transport](crates/interface/protocol/src/transport.rs), [WebSocket](crates/interface/protocol/src/websocket.rs) | [client](crates/interface/entelechy/tests/protocol.rs), [live agents](crates/interface/entelechy/tests/agent.rs), [what agents found wanting](crates/interface/entelechy/tests/findings/), [L7 AI bridge](bridges/ai/test.mjs) |
| Command line | [entelechy](crates/interface/entelechy/src/main.rs), [replay](crates/interface/session/src/replay.rs), [render](crates/interface/entelechy/src/render.rs), [packages](crates/interface/entelechy/src/package.rs), [scaffold](crates/interface/entelechy/src/scaffold.rs) | every example replay on three backends |
| C ABI | [ffi](crates/interface/ffi/src/lib.rs), [header](crates/interface/ffi/include/entelechy.h) | [ABI](crates/interface/ffi/tests/abi.rs), [header held to the exports](crates/interface/ffi/tests/header.rs), [C example](bridges/c/hello.c) |
| Player | [play](crates/interface/session/src/play/mod.rs) (`0 Hz` quiescent snapshot via `Play::resting` / `entelechy_play_resting`, centered Living Seed boot card via [hud](crates/interface/session/src/play/hud.rs), and zero-terminal on-canvas fault banner), shared by every front, each output given to the Host [device](crates/interface/documents/src/devices.rs) the scene names ([device and pointer tests](crates/interface/session/tests/pointer.rs)); shared native [window](crates/interface/player/src/window.rs) for both the live [Player](crates/interface/player/src/main.rs) (zero-arg empty stage, `Cmd+O` and drag-and-drop hot scene/replay switching over `Warm`) and the bundle-only standalone [Runner](crates/interface/player/src/runner.rs) (turn-by-turn shader pre-warm with Living Seed progress card), [audio](crates/interface/player/src/audio.rs), [gamepad](crates/interface/player/src/gamepad.rs); page [player.js](bridges/web/player.js) and [single.js](bridges/web/single.js) over [`entelechy_play_*`](crates/interface/ffi/src/play.rs) | [play tests](crates/interface/session/tests/play.rs), [usage test](crates/interface/player/tests/usage.rs), [runner tests](crates/interface/player/tests/runner.rs), [ABI test](crates/interface/ffi/tests/abi.rs) |
| Import and export | [import](crates/interface/import/src/), [export](crates/interface/export/src/) | [MIDI](crates/interface/formats/tests/midi.rs), [round trip](crates/interface/export/tests/round_trip.rs), [font round trip](crates/interface/export/tests/font.rs), [a scene as a bundle or one page](crates/interface/export/src/page.rs) ([gathered and packed](crates/interface/export/tests/page.rs), [its template](bridges/web/single.html), [opened from the disk in a browser](bridges/web/page.test.mjs)), [native desktop and mobile exports](crates/interface/export/src/native.rs) ([`.app`, `.exe`, `.ios` and `.android` tests](crates/interface/export/tests/native.rs), [iOS bridge](bridges/ios/App.m), [Android bridge](bridges/android/app/src/main/cpp/main.c)), [SVG](crates/interface/import/src/svg.rs), [vector replays](examples/vector/tests/) |
| Bridges | [bridges](bridges/) | [bridge tests](crates/interface/entelechy/tests/bridges.rs), when the program is present |

## Libraries

Packages built from existing modes, with no engine feature of their own.

| Library | Built from | Evidence |
| --- | --- | --- |
| [Stream compaction](packages/parallel.entpkg/kernels/compact.ent) | `prefix with sum`, reduce, scatter, `value(port)` | [culled replay](examples/culled/tests/culled.replay) |
| [Radix sort](packages/parallel.entpkg/kernels/sort.ent) | prefix and scatter | [neighbors replay](examples/swarm/tests/neighbors.replay) |
| [Mip levels](packages/image.entpkg/graphs/mipmap.esn) | `times`, `carry`, the rasterizer's `uv` | [mipmap replay](examples/mipmap/tests/ground.replay) |
| [Text](packages/text.entpkg/kernels/glyph.ent) | `Rows<Glyph>`, a `Grid<Float>` atlas | [text replay](examples/text/tests/text.replay) |
| [Skinning](packages/animation.entpkg/kernels/skin.ent), [rig contract](docs/RIGS.md) | maps over rows, `<~` | [rig tests](crates/interface/entelechy/tests/rigs.rs); [skin](examples/animation/tests/blend.replay), [crowd](examples/animation/tests/crowd.replay), [reach](examples/animation/tests/aim.replay), [elbow](packages/animation.entpkg/tests/aim.replay), [stride](examples/animation/tests/stride.replay) replays |
| [Rasterization](packages/raster.entpkg/kernels/triangle.ent) | scatter `min` of `Long` keys with `emit` | [edges](examples/mesh/tests/edges.replay) and [cutout](examples/mesh/tests/cutout.replay) replays on three backends, [session test](crates/interface/entelechy/tests/session.rs); [ten million triangles](examples/mesh/tests/ten_million.esn) for `bench` |
| [Ray tracing](packages/geometry.entpkg/kernels/trace.ent) | Moller-Trumbore maps over `Rows<TraceRay>`, accelerated by `native.trace@1` and `native.trace@2` | [trace replay](packages/geometry.entpkg/tests/trace.replay) on three backends, [GPU hardware RT & BVH parity](crates/backends/gpu/tests/parity.rs) |
| [Vector drawing](packages/vector.entpkg/kernels/fill.ent) | `prefix with sum`, scatter `max` with `emit`, `value(port)` | [vector replays](examples/vector/tests/) on three backends |
| [Rigid bodies](packages/physics.entpkg/kernels/solve.ent) | the math package, scatter `sum` of pushes, `times` with two `carry`s, `<~` | [physics replays](examples/physics/tests/) on three backends |
| [Widgets and letters](packages/ui.entpkg/README.md) | maps over grids, `<~` navigation state, `value(port)` and `count(port)` extents | [ui replay](examples/ui/tests/ui.replay) on three backends |
| [Font import](crates/interface/import/src/font.rs) | glyph outlines as `Rows` of segments | [importer tests](crates/interface/import/src/font.rs), on a font the tests build from the format's tables |
| [Image](packages/image.entpkg/package.esn) | maps, scatter `sum` and prefix `sum` (histograms), `times` with a carried step (jump flooding) | [image replay](packages/image.entpkg/tests/operations.replay) on three backends |
| [Noise](packages/random.entpkg/kernels/noise.ent) | functions over `Int` bits (`mul_wrap`, `>>>`) | [noise replay](packages/random.entpkg/tests/noise.replay) on three backends |
| [Camera finish](examples/nebula/graphs/finish.esn) | maps over grids, `size(port)` domains, the math package | [nebula replay](examples/nebula/tests/nebula.replay) on three backends |
| [Ways over a grid](packages/navigation.entpkg/kernels/grid_path.ent) | `times` with a carried pass over a cost grid | [skirmish replay](examples/slg/tests/skirmish.replay) on three backends |
| [Synthesis](packages/audio.entpkg/kernels/clock.ent) | a clock carried by `<~`, `rate()`, functions of time | [skirmish replay](examples/slg/tests/skirmish.replay) on three backends |

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
- Narrow storage (`as unorm8`, `as f16`): one rule in the Host
  ([encoding](crates/core/host/src/encoding.rs)), called by the interpreter,
  emitted by the JIT and the GPU; only a map or scan writes one, and a
  narrow link is not fused ([validator](crates/core/ir/src/validate.rs),
  [planner](crates/core/graph/src/evaluate/plan.rs)); `values()` are a
  Kernel's words, `storage()` the Field's bytes. On GPU adapters offering
  `SHADER_F16`, pure-`f16` unpacked/untextured storage buffers emit
  `enable f16;` with native `array<f16>` / `array<vec2<f16>>` /
  `array<vec4<f16>>` bindings, eliminating `clear_buffer` and `atomicOr`
  on `Float as f16` and `Vec3 as f16` outputs; non-`SHADER_F16` adapters
  use core WGSL `pack2x16float` / `unpack2x16float`
  ([storage](crates/backends/gpu/src/wgsl/storage.rs),
  [GPU parity](crates/backends/gpu/tests/parity.rs)). Held by the matrix's
  `narrow` rows and the [packed part](examples/data/README.md).
- Hardware block-compressed textures (`BC1`, `BC4`, `BC5`, `BC7`) and mipmap
  chains: pure-Rust Khronos §17 codec with multi-mip generation and BC5
  normal-map renormalization in the Host ([block](crates/core/host/src/block.rs)),
  deterministic CPU/JIT mipmap pyramid & 16x anisotropic reference filter
  ([mip](crates/core/host/src/mip.rs)), multi-mip zero-decompression `.ktx2`
  and `.dds` containers in [formats](crates/interface/formats/src/texture.rs)
  and [import](crates/interface/import/src/texture.rs), native
  `TEXTURE_COMPRESSION_BC` multi-mip 2D and `Texture2DArray` residency with
  multi-mip GPU-to-GPU compressed layer `join`, 1-pass direct `textureStore`
  GPU compute 2D/2D-array/3D mipmap chain generation (`MipGenPipelines`,
  writing all mip levels directly into core-WebGPU storage-writable formats
  `Rgba8Unorm`/`Rgba8Snorm`/`Rgba16Float`/`Rgba32Float`/`R32Float` in one
  compute pass with zero staging buffer or per-level copies, retaining the
  staging path for non-storage formats), hardware trilinear & 16x anisotropic
  samplers (`textureSampleLevel` / `textureSampleGrad`, optional
  `FLOAT32_FILTERABLE` with software mip fallback)
  ([texture](crates/backends/gpu/src/texture.rs),
  [wgsl](crates/backends/gpu/src/wgsl/prelude.rs),
  [gpu parity](crates/backends/gpu/tests/parity.rs),
  [jit parity](crates/backends/jit/tests/parity.rs)), plus bounded
  cancellable background `IoPool` and frame-paced `Volume<E>` `fields` node
  streaming ([store](crates/interface/documents/src/store.rs),
  [field_nodes](crates/interface/documents/src/inputs/field_nodes.rs),
  [fields tests](crates/interface/session/tests/fields_node.rs)).
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
| map | per element; the one mode with several outputs | [lowering](crates/frontend/compiler/src/lower/mod.rs), [WGSL](crates/backends/gpu/src/wgsl/map.rs), [GLSL ES 3.00](crates/backends/gpu/src/glsl/emitter.rs) |
| reduce (`merge`) | a fixed pairwise tree whatever the thread count; 256-lane subgroup/workgroup 8-level fold on GPU with giant-struct fallback | [reduce](crates/core/rules/src/reduce.rs), [WGSL](crates/backends/gpu/src/wgsl/reduce.rs), [JIT parity](crates/backends/jit/tests/parity.rs) |
| scan (`scan over … with`) | element order; memory copied, committed only on success | [rules](crates/core/rules/src/lib.rs) |
| scatter (`scatter over … into … with`) | as if applied in source order; outside addresses drop; on GPU, emitting scatters whose 1D `domain` derives from a device-resident `Int` Field (`value(port)`, `max`, linear scaling) dispatch via `dispatch_workgroups_indirect` with zero mid-frame Host readback and deferred zero-row validation | [WGSL](crates/backends/gpu/src/wgsl/scatter.rs), [indirect](crates/backends/gpu/src/indirect.rs), [domain](crates/core/graph/src/domain.rs), [parity](crates/backends/gpu/tests/parity.rs) |
| prefix (`prefix over … with`) | integer `sum`, `min`, `max` only; coalesced 256-lane hierarchical subgroup/workgroup scan on GPU | [block](crates/core/rules/src/lib.rs), [WGSL](crates/backends/gpu/src/wgsl/prefix.rs), [JIT](crates/backends/jit/src/lib.rs) |
| group (`group N`) | lanes meet at `exchange`, reached alike; single-wave register-only `subgroupShuffle` on GPU when `N <= min_subgroup_size`, workgroup shared memory otherwise; off the device, each stretch run lane by lane ([lanes](crates/core/ir/src/lanes.rs)) | [check](crates/core/ir/src/group.rs), [interpreter](crates/backends/cpu/src/lib.rs), [JIT parity](crates/backends/jit/tests/parity.rs), [GPU parity](crates/backends/gpu/tests/parity.rs) |

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
reachable GPU Kernels as they publish (sliced across turns with a
bounded warm-ahead window on a page: [turns](crates/interface/session/tests/turns.rs)), a replacement before it is ready where the
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
  `fuse_indexed` (a producer called at the rows its reader reads:
  `offset_read`, `row_reads`), `fuse_stencil` (a loop-free 2D `Grid`
  producer read within a sound interval-analyzed halo radius `1..=2`
  via `stencil_radius` / `stencil_radius_with_peers`, emitted as an
  ordinary `#tile_r{R}_{k}(pos, dom)` IR helper that the GPU backend lowers
  to cooperative `16x16`/`8x8` `var<workgroup>` shared-memory tile caching
  with a uniform `workgroupBarrier()` within a `14 KiB` core-WebGPU budget
  in [map](crates/backends/gpu/src/wgsl/map.rs)), and `counted` (an input
  only counted);
  recognizes self-coordinate `at(grid_or_volume, coord())` reads as
  pointwise, fuses across all eight narrow storage encodings (`f16`,
  `unorm8`, `unorm16`, `snorm8`, `u8`, `i8`, `u16`, `i16`) by inserting
  bit-exact non-failing IR quantization (`#quantize_f16`, emitted as a
  single `f32(f16(a0))` instruction on `SHADER_F16` GPUs), and proves
  equal dynamic extents across `size(port)` chains in the
  [planner](crates/core/graph/src/evaluate/plan.rs); calls a producer
  read at rows while the rows asked for are no more than it would make
  ([decision](docs/DECISIONS.md#work-is-spared-only-where-the-answer-is-the-same)),
  and stops fusing past node, instruction and carried-word bounds
  ([composition](crates/core/graph/tests/composition.rs));
  bit-identical ([test](crates/backends/cpu/tests/fusion.rs), on every
  backend [offset](crates/interface/session/tests/offset.rs), GPU tiled
  stencil parity in [parity](crates/backends/gpu/tests/parity.rs)). A reduce,
  scan, scatter, prefix, group, async or native node is never fused.
- A node placed as many times as a row says: `TimesRow` in the
  [graph](crates/core/graph/src/lib.rs), read in `expand_times`
  ([scene](crates/frontend/scene/src/lib.rs)), kept by the
  [documents](crates/interface/documents/src/scene.rs) and followed
  frame by frame by the [session](crates/interface/session/src/session.rs)
  ([test](crates/interface/session/tests/counted.rs)).
- One Field per declared output or the execution fails
  ([`held_to`](crates/core/kernel/src/lib.rs), [test](crates/core/kernel/tests/swap.rs)).
- Repetition: one Kernel over its own output is one group, one GPU
  encoder ([`Backend::execute_repeated`](crates/core/rules/src/backend.rs)).
- Native operations are versions at generation zero
  ([`NativeOp`](crates/core/rules/src/backend.rs)). The rasterizer
  (`native.rasterize@1` plain, `@2` cutout, `@3` canvas-sized, `@4`
  cutout canvas-sized, and `native.cover@1`) skips a triangle with any
  `w <= 0` whole
  ([rules](crates/core/rules/src/raster.rs), [near-plane replay](packages/raster.entpkg/tests/near.replay));
  its ports are read by field ([native replay](examples/mesh/tests/native.replay)).
  On the GPU ([raster](crates/backends/gpu/src/raster.rs),
  [Hi-Z](crates/backends/gpu/src/raster_hiz.rs)), dense opaque and
  alpha-cutout calls run a zero-redraw 2-phase conservative MAX-Z Hi-Z
  pipeline (non-transparent macro-occluders draw once directly into
  `places` with global row + 1 and `depth`, seeding both the
  odd-dimension-safe MAX-Z Hi-Z pyramid and Phase 4 hardware early-Z via
  `LoadOp::Load` while recording a Phase-1 `alive` bitmask in the second
  half of `marks` so Phase 3 skips culled and Phase-1 triangles with a
  single bit test; `native.cover@1` seeds from prior canvas depth),
  sample cutout `mask` inputs through a lazily cached 1-channel `R32Float`
  alpha-only 2D tiled texture (4x lighter in texture cache than
  `Rgba32Float`, extracted once alongside the conservative `Rgba8Unorm`
  MIN/MAX alpha pyramid `AlphaMinMaxPyramid` in a single compute pass
  without allocating a throwaway 1-mip color mirror; sampled via
  `textureSampleLevel` when `FLOAT32_FILTERABLE` is available and tiled
  `textureLoad` otherwise, culling 100% transparent triangles and tagging
  provably opaque micro-triangles in `small_rows` to bypass `mask_alpha`),
  load triangle clip words once per sort with row-hoisted micro-triangle
  edge steps, apply exact 1-pixel-box coverage and depth rejection before
  compaction, dispatch surviving micro-triangles via
  `dispatch_workgroups_indirect` with Phase-1 `depth_tex` early-Z for cutout
  triangles, and resolve each pixel with a single `fragment()` evaluation.
  The ray tracer (`native.trace@1` single-mesh, `native.trace@2` instanced
  multi-mesh) builds pure-Rust SAH BVHs on CPU/JIT
  ([rules](crates/core/rules/src/trace.rs)) and on GPU
  ([gpu trace](crates/backends/gpu/src/trace.rs),
  [gpu LBVH](crates/backends/gpu/src/trace_lbvh.rs)) executes batched,
  GPU-timed hardware ray queries (`EXPERIMENTAL_RAY_QUERY`, with automatic
  double-buffered `DynamicHwBlas` and `DynamicHwTlas` promotion for
  deforming meshes and animating instances, same-frame `DynamicHwTlas`
  slot reuse before any Host readback, and hardware-RT-only Host routing
  + scheduler hoisting/async prefetch for `meshes`/`instances` producers
  via `Backend::needs_host_trace_instances`) or a zero-readback full-GPU
  LBVH builder (30-bit Morton + 8-pass GPU radix sort + Karras binary
  radix tree + race-free 2-tier in-place dynamic AABB refit, multi-mesh
  BLAS slicing via `encode_slice_into`, and full-GPU dynamic TLAS LBVH
  `GpuLbvhTlas`) with GPU-driven active-ray stream compaction and
  `dispatch_workgroups_indirect` (including 11-storage-binding compaction
  for `native.trace@2` when `max_storage_buffers_per_shader_stage >= 11`,
  keeping the 10-binding baseline for WebGPU)
  ([trace replay](packages/geometry.entpkg/tests/trace.replay),
  [GPU parity](crates/backends/gpu/tests/parity.rs)).

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
- `use` across files and cycles ([sources](crates/interface/documents/src/sources.rs));
  the names each document sees, shadowing and duplicate names
  ([scope](crates/frontend/language/src/scope.rs), [session test](crates/interface/entelechy/tests/session.rs),
  [the contract case by case](crates/interface/session/tests/scope.rs)).
- Parameters of any value type, checked by layout
  ([`Param`, `value_as`](crates/core/ir/src/lib.rs), [`Value`](crates/core/abi/src/lib.rs),
  [`set_param`](crates/interface/documents/src/scene.rs),
  [bind](crates/core/graph/src/evaluate/bind.rs),
  [GPU slots](crates/backends/gpu/src/wgsl/storage.rs); [cube](examples/image/README.md)).

### Domain

- A node's `domain` expression: [graph](crates/core/graph/src/domain.rs),
  [scene](crates/frontend/scene/src/domain.rs); printed back so documents
  round-trip.
- `value(port)` reads a row of scalars, bounded by `VALUE_MAX_ELEMENTS`
  ([`domain_values`](crates/core/graph/src/evaluate/bind.rs)).
  On device backends, [`Graph::schedule_for_device`](crates/core/graph/src/lib.rs)
  hoists `value(port)` count producers ahead of independent work and defers
  `value(port)` consumers so async readbacks overlap GPU execution while
  CPU/JIT sessions keep canonical [`Graph::order`](crates/core/graph/src/lib.rs)
  ([composition](crates/core/graph/tests/composition.rs)); emitting
  `Mode::Scatter` nodes whose 1D count derives from a GPU-resident `Int`
  Field bypass Host readback completely via `dispatch_workgroups_indirect`
  ([indirect](crates/backends/gpu/src/indirect.rs)).
- All five GPU modes (`map`, `group`, `reduce`, `prefix`, `scatter`) batch
  into a single command encoder with 3-slot ringed uniform, small-upload, and
  failure-readback slabs ([batch](crates/backends/gpu/src/batch.rs),
  [failure](crates/backends/gpu/src/failure.rs),
  [modes](crates/backends/gpu/src/modes.rs),
  [later](crates/backends/gpu/src/later.rs)): per-node failure records and
  unclamped indirect-domain checks pack into one storage-aligned slab copied
  once per batch and mapped immediately at submit time on both native and
  browser WebGPU (`Wait::parks_with_host`) targets (preserving 2-frame
  overlap and letting `Submitted::wait` skip redundant waits once the batch
  failure slab has resolved), `Pipeline.main` binds context, params, and failure
  storage with dynamic offsets backed by an evaluation-aged `BindGroupCache`
  and view-preserving `SpareMirrors` ([pool](crates/backends/gpu/src/pool.rs),
  [texture](crates/backends/gpu/src/texture.rs)), plain non-mipmapped
  non-loop 2D `at_linear` inputs produced fresh in the current evaluation
  (`created_eval == now`) bind directly as zero-copy L2-cached storage
  buffers via a lazy dynamic pipeline variant ([prepare](crates/backends/gpu/src/prepare.rs))
  while static `ONCE`, Host-uploaded, mipmapped, volume, and loop-sampled
  textures stay hardware-mirrored, and the background `Waiter`
  defers polling while passes encode and backs off when idle
  ([wait](crates/backends/gpu/src/wait.rs)). [`Runtime::sized_off`](crates/core/kernel/src/lib.rs)
  guards un-prefetched GPU-resident inputs from pulling small runs back to
  CPU JIT mid-batch.
- Window size and scale are part of a node's cache key.

### The language

- Sugar the parser and lowering write out
  ([parser](crates/frontend/language/src/parser.rs), [lowering](crates/frontend/compiler/src/lower/mod.rs)).
- Formula builtins (`dot`, `normalize`, `inverse`, …) are lowered to
  arithmetic, so every backend computes one formula
  ([composite](crates/frontend/compiler/src/lower/composite.rs), [matrix](crates/frontend/compiler/src/lower/matrix.rs),
  [WGSL `select`](crates/backends/gpu/src/wgsl/emitter.rs)).
- Work bounds ([decision](docs/DECISIONS.md#the-host-bounds-work-and-memory)): `Manifest::budget`
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
  or input change ([composition](crates/core/graph/tests/composition.rs)).
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
  moves it to a worker ([hot swap](crates/core/kernel/tests/swap.rs)).
- A sync compiles every new or changed Kernel of the documents it read,
  and prepares the new ones, on every core at once through a
  [`Preparer`](crates/core/kernel/src/lib.rs); each document then
  publishes in order as if alone ([prime](crates/interface/session/tests/prime.rs)).
  `entelechy check` checks the documents a scene uses the same way, each on
  its own, and says what they found in their order
  ([spread](crates/interface/documents/src/spread.rs)).
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
  work another way ([decision](docs/DECISIONS.md#a-platform-is-the-hosts-world-what-a-scene-computes-is-not), [later](crates/core/kernel/src/later.rs),
  [store](crates/interface/documents/src/store.rs), [clock](crates/core/kernel/src/clock.rs),
  [bridges test](crates/interface/entelechy/tests/bridges.rs)).

### Measuring

- A profile line subtracts `Backend::measurement_floor`
  ([runtime](crates/core/kernel/src/lib.rs), [bench](crates/interface/session/src/bench.rs)).
- The GPU buffer pile serves the smallest buffer at most twice the size
  ([`Spare::take`](crates/backends/gpu/src/pool.rs)).
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
  A NaN already in a buffer is tested on its `bitcast<u32>` words rather
  than `x != x`, and `pow` goes through `ent_pow` where WGSL leaves negative
  bases undefined ([prelude](crates/backends/gpu/src/wgsl/prelude.rs),
  [parity](crates/backends/gpu/tests/parity.rs)).
- Narrow `unorm8` outputs and non-`SHADER_F16` `f16` outputs sharing a
  32-bit word are `atomic<u32>` stored by `atomicOr`, with core WGSL
  `pack2x16float` / `unpack2x16float` for fallback `f16` and integer
  bit-packing for `unorm8`; when `SHADER_F16` is available, pure-`f16`
  unpacked/untextured outputs bind native `array<f16>` / `array<vec2<f16>>`
  / `array<vec4<f16>>` directly without `atomic<u32>` or `clear_buffer`
  ([wgsl](crates/backends/gpu/src/wgsl/storage.rs) `narrow_store_body`).
- wgpu's error scopes are thread-local only with its `std` feature, which
  the GPU backend enables.
- Uniforms are rewritten per node with `queue.write_buffer`, which orders
  against submissions: a frame is not one submission. On a page
  (`Wait::parks_with_host`), unsettled submissions are drained periodically
  (`PageQueue`) and large or deep-volume `Map` runs slice across `ctx[5].w`
  so startup baking bursts never trip Windows WDDM's 2-second TDR watchdog.
