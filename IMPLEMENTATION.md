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

- The sound card is a device adapter in the Player, not a runtime concept
  ([audio](crates/player/src/audio.rs)): it claims the first graph output
  that is rows of one or two floats, takes each evaluation's block into a
  lock-free ring, and feeds the device from the ring in its real-time
  callback, which never waits on the Graph or the GPU and never
  allocates; a dry ring plays silence and counts an underrun. Materializing
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

- A node's `domain` may be an expression over its inputs and parameters
  ([domain](crates/graph/src/domain.rs)): a small arithmetic language with
  `size(port)` and `count(port)`, accepted at document lowering, resolved
  when the node binds (and at planning, so fusion and `orla graph` see
  the extent), and printed back so documents round-trip. It lets a Kernel
  that changes topology size its output from its input, as the
  [mesh example](examples/mesh/README.md)'s subdivision does, and the
  [tiles example](examples/tiles/README.md) draw a map at its tile size.

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
- Input is context, like time: `pointer()`, `pointer_down()`, and `key(code)`
  behind `input.read`; only nodes that declare it re-execute when input
  changes ([ABI](crates/abi/src/lib.rs), [parity](crates/gpu/tests/parity.rs)).
- Application tests are data: `.replay` files script frames, pointer, keys,
  reset, and pixel expectations; `orla replay` runs them and the interface
  tests run every `examples/*/tests/*.replay`
  ([replay](crates/cli/src/replay.rs)).

- Structs are Kernel-internal record types: declared in `.orla`, indexed in
  [the IR struct table](crates/ir/src/lib.rs), built and updated with
  `Name { .. }` literals, and allowed wherever a value is (loops carry them,
  functions take and return them). They never cross the ABI as parameters or
  ports. The CPU backend flattens them to slot ranges
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
  [execution tests](crates/cpu/tests/execute.rs). `time()` needs `time.read`;
  only nodes that declare it re-execute when time advances
  ([clock caching test](crates/graph/tests/composition.rs)).
- The CPU backend computes rows in parallel and writes sequentially; results
  and error attribution match a single-threaded run
  ([parallel rows test](crates/cpu/tests/execute.rs)).
- Undeclared capabilities are compile-time diagnostics with positions
  ([compile tests](crates/compiler/tests/compile.rs)); the IR validator remains
  the backstop.

Use [USAGE](docs/USAGE.md) for reproducible commands and [HANDOFF](HANDOFF.md) for
the next action. Git preserves completed work. Do not maintain feature
inventories, test counts, or completion percentages here.
