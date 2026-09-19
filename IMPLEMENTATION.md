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
| Orla IR | [ir](crates/ir/src/lib.rs), [validation](crates/ir/src/validate.rs), [display](crates/ir/src/display.rs) | Unit tests: typing, SSA order, capability coverage |
| Kernel Language | [language](crates/language/src/lib.rs) | Parser tests: syntax, precedence, positioned errors |
| Compiler | [compiler](crates/compiler/src/lib.rs), [lowering](crates/compiler/src/lower.rs) | [compile tests](crates/compiler/tests/compile.rs) |
| CPU backend | [cpu](crates/cpu/src/lib.rs), [interpreter](crates/cpu/src/eval.rs) | [execution tests](crates/cpu/tests/execute.rs) |
| Kernel Runtime | [runtime](crates/runtime/src/lib.rs) | [hot swap tests](crates/runtime/tests/hot_swap.rs) |
| GPU backend | [gpu](crates/gpu/src/lib.rs), [WGSL emission](crates/gpu/src/wgsl.rs) | [CPU/GPU parity tests](crates/gpu/tests/parity.rs) (skip without an adapter) |
| Graph Runtime | [graph](crates/graph/src/lib.rs), [document](crates/graph/src/document.rs), [evaluator](crates/graph/src/evaluate.rs) | [composition tests](crates/graph/tests/composition.rs) |
| Interface | [cli](crates/cli/src/main.rs), [session](crates/cli/src/session.rs), [ppm](crates/cli/src/ppm.rs) | [session and binary tests](crates/cli/tests/session.rs) |
| Player | [window loop](crates/player/src/main.rs), [presentation](crates/player/src/lib.rs) | Presentation unit tests; [material ball](examples/material-ball/README.md) for the live loop |

## Contracts worth knowing

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
  there), so `publish` is a pointer swap. The GPU backend keeps Host ownership
  of images by uploading inputs and reading outputs back per execution; GPU
  and CPU results agree within float tolerance, and integer arithmetic
  failures are reported only by the CPU backend.

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
