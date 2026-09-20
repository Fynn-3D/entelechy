# Implementation evidence

Code and executable tests establish what Orla supports. This index connects the
[roadmap](Orla%20Architecture%20Plan.md) to places to inspect; it does not certify
that a phase is complete. Read the assertions and run the checks before making a
capability claim.

## The four proofs

| Proof | Where it is exercised |
| --- | --- |
| `.orla` compiles | [compiler tests](crates/compiler/tests/compile.rs) |
| Kernel calls Host only through ABI | [scope tests](crates/abi/src/lib.rs), [denied execution](crates/cpu/tests/execute.rs), [validator capability checks](crates/ir/src/validate.rs) |
| Kernel hot-swaps without restart | [hot swap tests](crates/runtime/tests/hot_swap.rs), [session tests](crates/cli/tests/session.rs) |
| Graph references KernelID | [composition tests](crates/graph/tests/composition.rs), [example scene](examples/grayscale/scene.osn) |

## Layers

| Layer | Implementation | Executable evidence |
| --- | --- | --- |
| Host Core | [host](crates/host/src/lib.rs) | Unit tests: handles, budgets, bounds |
| Capability ABI | [abi](crates/abi/src/lib.rs) | Unit tests: grant checks before Host access |
| Orla IR | [ir](crates/ir/src/lib.rs), [validation](crates/ir/src/validate.rs), [display](crates/ir/src/display.rs), [fusion](crates/ir/src/fuse.rs) | Unit tests: typing, SSA order, capability coverage; [fusion bit-identity](crates/ir/tests/fusion.rs) |
| Kernel Language | [language](crates/language/src/lib.rs) | Parser tests: syntax, precedence, positioned errors |
| Compiler | [compiler](crates/compiler/src/lib.rs), [lowering](crates/compiler/src/lower.rs) | [compile tests](crates/compiler/tests/compile.rs) |
| CPU backend | [cpu](crates/cpu/src/lib.rs), [interpreter](crates/cpu/src/eval.rs) | [execution tests](crates/cpu/tests/execute.rs) |
| Kernel Runtime | [runtime](crates/runtime/src/lib.rs) | [hot swap tests](crates/runtime/tests/hot_swap.rs) |
| GPU backend | [gpu](crates/gpu/src/lib.rs), [WGSL emission](crates/gpu/src/wgsl.rs) | [CPU/GPU parity tests](crates/gpu/tests/parity.rs) (skip without an adapter) |
| Graph Runtime | [graph](crates/graph/src/lib.rs), [document](crates/graph/src/document.rs), [evaluator](crates/graph/src/evaluate.rs) | [composition tests](crates/graph/tests/composition.rs) |
| Interface | [cli](crates/cli/src/main.rs), [session](crates/cli/src/session.rs), [replay](crates/cli/src/replay.rs), [ppm](crates/cli/src/ppm.rs) | [session, binary, and example-replay tests](crates/cli/tests/session.rs) |
| Creation protocol | [protocol](crates/cli/src/protocol.rs), [contract](docs/PROTOCOL.md) | [end-to-end client test](crates/cli/tests/protocol.rs) |
| Player | [window loop](crates/player/src/main.rs), [CPU blit](crates/player/src/lib.rs), [device presenter](crates/gpu/src/present.rs) | Presentation unit tests, blit shader compile test; [material ball](examples/material-ball/README.md) for the live loop |

## Contracts worth knowing

- There is one resource, following [the Typed Field decision](Orla%20Typed%20Field%20Architecture.md):
  a Field of elements in one of three shapes, `Rows`, `Grid`, or `Volume`,
  with one 32-bit word per scalar ([host](crates/host/src/lib.rs)). Port
  types are structural (`element`, `shape`, `schema`); `Image` is the
  compiler's alias for `Grid<Vec4>` and `type` declares more aliases
  ([compiler](crates/compiler/src/lower.rs)). Both backends carry every
  element as words, so integers and booleans are exact and the two agree
  bit for bit on the examples ([execution test](crates/cpu/tests/execute.rs),
  [parity test](crates/gpu/tests/parity.rs)); the GPU dispatches over three
  axes and keeps every shape resident. Bindings are checked by shape and
  layout, never by spelling ([evaluator](crates/graph/src/evaluate.rs)).
  The [particles](examples/particles/README.md) (rows),
  [tiles](examples/tiles/README.md) (a grid of structs through an alias),
  [Life](examples/life/README.md) (a grid of booleans and integers), and
  [volume](examples/volume/README.md) examples are the working proofs.

- Presentation is an observation the GPU can do itself: the Player gives the
  GPU backend its window, and a resident output is drawn onto the surface by
  a blit shader that reads the image's storage buffer; the frame never
  crosses to the Host. The title reports the path taken and the download
  count. Without a GPU, or when the surface cannot be configured, the Player
  downloads and blits through softbuffer ([presenter](crates/gpu/src/present.rs)).

- Residency: the Host owns every image, but a slot may hold its pixels on a
  device as an opaque object the backend understands
  ([host](crates/host/src/lib.rs)). The GPU backend binds resident inputs
  directly, uploads Host-memory inputs once, and leaves outputs resident;
  committed state therefore ping-pongs on the device. Pixels return only at
  observation boundaries through [`Runtime::image`](crates/runtime/src/lib.rs)
  (export, protocol `read`, replay `expect`, the Player's present), which is
  where device work is waited for ([residency test](crates/gpu/tests/parity.rs)).

- Fusion is backend-neutral and exact: a producer read by exactly one
  consumer, only through `sample`, over the same domain, that is neither a
  graph output, a state source, nor pinned, is inlined into that consumer
  by [`fuse`](crates/ir/src/fuse.rs) and executed as one detached Kernel
  version. Results are bit-identical on the CPU
  ([IR test](crates/ir/tests/fusion.rs), [graph tests](crates/graph/tests/composition.rs));
  absorbed nodes are not materialized until pinned.

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

- Native operations are device access published as versions: the runtime
  registers each [`NativeOp`](crates/runtime/src/lib.rs) at generation
  zero with a manifest and a textual contract, refuses to publish a Kernel
  over it, and dispatches it to `Backend::execute_native`; the evaluator
  never fuses one. `native.rasterize@1` is the first: on the CPU a
  reference rasterizer ([raster](crates/cpu/src/raster.rs)), on the GPU a
  render pipeline into two float targets and a depth target compacted
  into the tight word layout ([raster](crates/gpu/src/raster.rs)); they
  agree on every interior pixel ([parity test](crates/gpu/tests/parity.rs),
  [runtime test](crates/runtime/tests/hot_swap.rs)). Projection and
  shading stay Kernels on either side of it
  ([mesh example](examples/mesh/README.md)).

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

- Reduction is the second execution mode, declared by a `merge(a, b)`
  block: the entry maps over the first input's domain and `merge`
  folds the results pairwise into one `Rows` element; the manifest
  carries `mode`, the module carries `merge`, and the validator checks
  the fold's signature against the output element
  ([IR](crates/ir/src/lib.rs), [validator](crates/ir/src/validate.rs),
  [compiler](crates/compiler/src/lower.rs)). The CPU folds level by level
  in a fixed tree regardless of thread count; the GPU maps into a scratch
  buffer and folds with a second pipeline in the same tree, so integers
  are exact on both and floats agree to rounding
  ([CPU](crates/cpu/src/lib.rs), [WGSL](crates/gpu/src/wgsl.rs),
  [parity test](crates/gpu/tests/parity.rs)). The evaluator gives a reduce
  node the domain of one element and keeps it out of fusion
  ([evaluator](crates/graph/src/evaluate.rs)). The
  [exposure example](examples/exposure/README.md) meters a frame and
  tone-maps by the result.

- Ordered work may own memory: `memory ram: Rows<Int>;` is a port that
  is an input and an output at once (last in both lists of the
  manifest, named in `memories`), which a scan's steps read with `at` as
  the earlier steps left it and write with `put(ram, address, value)` in
  step order, an address outside dropped. The CPU works on a copy of the
  bound Field and hands the copy back as the output; the GPU copies the
  input's buffer, binds the copy read-write, and adopts it as the output;
  the bound Field is never written, so a failed evaluation commits
  nothing, and both backends agree bit for bit
  ([IR](crates/ir/src/lib.rs), [validator](crates/ir/src/validate.rs),
  [compiler](crates/compiler/src/lower.rs), [CPU](crates/cpu/src/eval.rs),
  [WGSL](crates/gpu/src/wgsl.rs), [parity](crates/gpu/tests/parity.rs),
  [feedback test](crates/graph/tests/composition.rs)). With the integer
  bit operators, this is what a machine needs: random-access RAM written
  step by step, persisting across frames through `<~`; the
  [CHIP-8 example](examples/chip8/README.md) is one, every opcode, its
  replay reading the screen memory on both backends.
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

- Writing where a Kernel computes is the fourth execution mode, declared
  by `scatter into target with rule` in place of `compute`: the entry
  maps over the first input and yields an address and a value, and the
  output is the target with every value combined into the element at its
  address, as if the sources were applied one after another in element
  order. That order is the determinism rule: `sum`, `min`, `max`, and
  `last` come out the same on both backends, floats included, and an
  address outside the target is dropped ([IR](crates/ir/src/lib.rs),
  [validator](crates/ir/src/validate.rs), [compiler](crates/compiler/src/lower.rs)).
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
- A node's `domain` may be an expression over its inputs and parameters
  ([domain](crates/graph/src/domain.rs)): a small arithmetic language with
  `size(port)` and `count(port)`, accepted at document lowering, resolved
  when the node binds (and at planning, so fusion and `orla graph` see
  the extent), and printed back so documents round-trip. It lets a Kernel
  that changes topology size its output from its input, as the
  [mesh example](examples/mesh/README.md)'s subdivision does, and the
  [tiles example](examples/tiles/README.md) draw a map at its tile size.

- Sampling authority is a builtin gated by layout, not by the word
  `Image`: `at_linear(f, position)` interpolates between neighbouring
  elements of any input whose layout is all floats (a struct of floats
  included, mixed fieldwise), bilinear over grids and trilinear over
  volumes, with element centres at half-integers and clamped edges. The
  CPU sums weighted corners in words; the GPU emits one helper per input
  and one mixer per float struct ([interpreter](crates/cpu/src/eval.rs),
  [WGSL](crates/gpu/src/wgsl.rs), [parity test](crates/gpu/tests/parity.rs)).
  The [volume example](examples/volume/README.md) reads its cloud that way.

- Declarations cross files through `use`: a document's used files (and
  theirs) contribute their top-level structs and type aliases to every
  Kernel the document holds, resolved by the session and `check` through
  a `declare` callback the graph loader applies to the root and each
  sub-graph ([graph](crates/graph/src/document.rs),
  [session](crates/cli/src/session.rs)). A changed library republishes
  the Kernels that inherit from it and re-reads the root; two different
  declarations of one name are refused with both files named
  ([session test](crates/cli/tests/session.rs)).

- The creation protocol is an interface over the session, not a new
  authority: `prepare` never publishes, `publish` checks an expected
  generation, `compose` rewrites the scene document so the file stays the
  source of truth, and diagnostics carry line and column
  ([protocol](crates/cli/src/protocol.rs)).

- Documents: one grammar for `.orla` libraries and `.osn` graphs
  ([contract](docs/DOCUMENTS.md), [frontend](crates/language/src/lib.rs),
  [graph lowering](crates/graph/src/document.rs),
  [session loader](crates/cli/src/session.rs)). Nodes reference Kernels by
  id or carry them inline; the loader publishes inline Kernels and follows
  `use`. Proof: [parser tests](crates/language/src/parser.rs),
  [lowering tests](crates/graph/src/document.rs),
  [session tests](crates/cli/tests/session.rs), and the
  [self-contained material ball](examples/material-ball/scene.osn).

- State lives in the Graph, not in Kernels: a `port <~ node.output from
  initial` binding reads the source as committed by the previous successful
  evaluation. Every node of one evaluation reads the same committed state;
  the evaluation commits atomically at its end; a failed evaluation commits
  nothing; a republished Kernel inherits the state; `reset_state` returns to
  the initial sources ([evaluator](crates/graph/src/evaluate.rs),
  [feedback tests](crates/graph/tests/composition.rs),
  [Life example](examples/life/README.md)).
- Input is context, like time: `pointer()`, `pointer_down()`, `button(n)`,
  `wheel()`, and `key(code)` behind `input.read`; only nodes that declare it
  re-execute when input changes ([ABI](crates/abi/src/lib.rs),
  [parity](crates/gpu/tests/parity.rs)).
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
  ([library](examples/shared/text.orla), [font](examples/shared/font.orla),
  [text example](examples/text/README.md)). The atlas is baked by a
  Kernel through `run --save` and both backends draw it bit for bit
  ([replay](examples/text/tests/text.replay)).
- A node repeats with `times = N;` and `carry port <- output;`: the
  document lowering expands it into `name/1..N` chained through the
  carried ports, with `name.port` meaning the last; `N` may be a graph
  param, which the session then treats as structural and lowers the
  document again when it changes ([document](crates/graph/src/document.rs),
  [tests](crates/graph/src/document.rs), [session test](crates/cli/tests/session.rs)).
  The fluid's Jacobi solve, the SLG's relaxation passes and the mesh
  example's subdivision level are written that way.
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
  [its runner](bridges/web/test.mjs); bridges are clients outside the
  workspace.
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
- Functions are shared like structs: declared at a document's top level
  and inherited through `use`; one with a Field parameter is a template
  the compiler instantiates per bound port, so a grid walk or a staggered
  read is written once for every Kernel that needs it
  ([lowering](crates/compiler/src/lower.rs), [tests](crates/compiler/tests/compile.rs)).
- An evaluation can be demanded: `Evaluator::evaluate_for` runs the named
  nodes, what they read, every `<~` source, and never-produced outputs, and
  leaves the rest at its last value; a replay demands the picture only on
  frames a pixel expectation follows, so a replay of state costs what the
  state costs ([evaluator](crates/graph/src/evaluate.rs), [replay](crates/cli/src/replay.rs)).
- A Field comes in from outside through the protocol's `write`, which
  replaces a state port's committed value with words in its layout
  ([evaluator](crates/graph/src/evaluate.rs), [protocol](crates/cli/src/protocol.rs)).
- Application tests are data: `.replay` files script frames, pointer, keys,
  reset, and pixel expectations; `orla replay` runs them and the interface
  tests run every `examples/*/tests/*.replay`
  ([replay](crates/cli/src/replay.rs)).

- Structs are Kernel-internal record types: declared in `.orla`, indexed in
  [the IR struct table](crates/ir/src/lib.rs), built and updated with
  `Name { .. }` literals, and allowed wherever a value is (loops carry them,
  functions take and return them). They never cross the ABI as parameters or
  ports. The CPU backend lays them out as words, fields in order
  ([interpreter](crates/cpu/src/eval.rs)); the GPU backend emits native WGSL
  structs. Proof: [struct tests](crates/cpu/tests/execute.rs),
  [diagnostics](crates/compiler/tests/compile.rs),
  [parity](crates/gpu/tests/parity.rs), and the
  [ray-march example](examples/raymarch/kernels/march.orla).

- Backends implement [`Backend`](crates/runtime/src/lib.rs): `prepare` does
  the expensive work (the GPU backend compiles WGSL and builds the pipeline
  there), so `publish` is a pointer swap. GPU and CPU results agree within
  float tolerance (exactly on the examples, whose hashes avoid `sin` of
  large arguments), and integer arithmetic failures are reported only by
  the CPU backend.

- Control flow is structured and expression-shaped: `if c { a } else { b }`
  and `loop N with x = init until done { next }` lower to [`Op::If` and
  `Op::Loop`](crates/ir/src/lib.rs) with pre-order value ids and block
  scoping proved by [the validator](crates/ir/src/validate.rs). Only the taken
  branch executes and loop bounds are compile-time constants
  ([execution tests](crates/cpu/tests/execute.rs),
  [scoping diagnostics](crates/compiler/tests/compile.rs)). The
  [ray-march example](examples/raymarch/README.md) is the working proof.

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
- The CPU backend compiles a module to word-addressed code once per
  execution and runs it once per element: every value has a fixed word
  offset in the ABI's own layout, so arguments, struct fields, vector
  components and a loop's carried value are aliases that cost nothing,
  Field reads copy words out of slices the scope lent once, and nothing
  allocates per element ([interpreter](crates/cpu/src/eval.rs)). Rows run
  in parallel and are written sequentially; results and error attribution
  match a single-threaded run ([parallel rows test](crates/cpu/tests/execute.rs)),
  and the GPU parity tests and every example replay hold the two backends
  to the same pixels.
- Undeclared capabilities are compile-time diagnostics with positions
  ([compile tests](crates/compiler/tests/compile.rs)); the IR validator remains
  the backstop.

Use [USAGE](docs/USAGE.md) for reproducible commands and [HANDOFF](HANDOFF.md) for
the next action. Git preserves completed work. Do not maintain feature
inventories, test counts, or completion percentages here.
