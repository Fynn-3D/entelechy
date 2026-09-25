# Handoff

**Answers:** what this repository holds that nothing else records, and the next action.

## Context

| Fact | What follows from it |
| --- | --- |
| The interpreter is the reference; the JIT matches it bit for bit, the GPU to rounding. | Where three answers differ, `--backend cpu` is the right one. |

Verified surface, as of `main`:

| Where | How |
| --- | --- |
| macOS, Apple M4, Metal | every round |
| WebAssembly under WASI (`wasmtime`), the interpreter | the example replays, `serve` on standard input |
| A page: Orla as WebAssembly in headless Chrome, on its JIT (WebAssembly) and on WebGPU, files handed in | the web bridge's test, from `cargo test --workspace` (needs `wasm-bindgen` at Cargo.lock's version) |
| 32-bit WebAssembly under node: the core's, the front end's, the interpreter's and the JIT's tests, the JIT's code WebAssembly | `crates/interface/orla/tests/wasm.rs`, from `cargo test --workspace` |
| Interpreter, JIT, GPU | `examples/*/tests/*.replay` on all three |
| 36 language features × 7 execution forms (the modes, and a map of several outputs), and 8 narrow encodings read and written | `crates/backends/conformance`, JIT and GPU against the interpreter |
| 8 features of `Double` and `Long` × 6 modes; 5 of `Long` alone in maps and reductions | the same: the JIT against the interpreter, the GPU for `Long` |
| Keyboard, pointer | by hand |
| Pad sticks, buttons, triggers | by replay (`stick`, `button` lines) |
| Random programs: the JIT bit for bit, the GPU's integers and booleans exactly | `crates/backends/{jit,gpu}/tests/fuzz.rs`, from `orla_conformance::random`; `ORLA_FUZZ=<n>` runs more |
| Mutated documents and protocol requests never crash or hang the front end or a session | `crates/interface/orla/tests/hostile.rs` |
| A session's random life: a success advances by one, a failure changes nothing, the Host does not grow | `crates/interface/session/tests/properties.rs` |
| Blender 5.2, headless Chrome, Godot 4.6, C ABI | `cargo test --workspace`, each skipped when its tool is absent |
| A live Blender | its MCP add-on, `127.0.0.1:9876`, `{"type":"execute","code":…,"strict_json":false}` + `\0` |

## Next

Converge the code on the architecture, one round at a time, each on its
own branch and merged on the user's word. The evaluation behind it: the
soul holds (hot reload, capabilities, a backend-neutral IR, KernelIDs,
one Field model, layering held by `layers.rs`); what drifts is
behaviour each interface decides for itself, and IR knowledge each
backend derives again.

1. Bugs and holes. One `node.port` parser, split at the last dot, for
   every caller (replay's `reset` and `expect count` split at the
   first, so a placed node's dotted id breaks). `save` and `export` go
   through the session's store and write allowance, not around them.
2. The IR owns what every backend repeats: a function's result type,
   a walk over an op's blocks; the JIT's failure codes and layout shared
   by its native and WebAssembly code, named.
3. Thin interfaces: the session reads a Field or a picture back, so a
   front no longer reaches into the runtime; one rule for the picture
   and the sound an output is; one way to open a session (backend,
   fallback, grants); `check` in the session; no backend types in the
   session's API. Whether the frame loops become one is decided there.
4. God files: the GPU backend's `lib.rs` and `wgsl.rs` split by mode,
   named alike; the scene's `Lowering::level` split into phases.
5. God objects: the runtime's background preparation and telemetry
   their own types; the evaluator's handle ledger a type whose fields
   only it touches.
6. Odds: codecs in `orla-formats`, one home for name suggestions, the
   `.osn` printer beside its parser, unused dependencies, AGENTS and
   `layers.rs` saying the same about backend tests.

Afterwards the user chooses among: Safari 27's WebGPU, once this machine
has it; a second native platform.
