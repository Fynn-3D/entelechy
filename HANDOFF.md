# Handoff

**Answers:** what this repository holds that nothing else records, and the next action.

## Context

| Fact | What follows from it |
| --- | --- |
| The interpreter is the reference; the JIT matches it bit for bit, the GPU to rounding. | Where three answers differ, `--backend cpu` is the right one. |
| A new Kernel's cost on the GPU is its pipeline's compile: 40 to 120 ms for the voxel example's Kernels on Metal, nearly all of preparing it; one compiled before comes from the driver's cache in under a millisecond. | Natively it is paid off the frame (the preparation worker). A page asks its browser for each pipeline as the Kernel prepares (`createComputePipelineAsync`, through `js-sys` and `wasm-bindgen` in `entelechy-gpu`), compiled off the thread that answers the device; a frame that needs one not yet compiled waits for it. Data, params and structure (`write`, `param`, a `compose` that changes no Kernel) never compile. Scenes are built in the native Player; a page shows them. |
| The session holds the world (scene, ledger, sources, history, inputs, writes) and also opens backends; only its `lib.rs`, `session.rs` and `observe.rs` may name one (`crates/interface/entelechy/tests/layers.rs`). | A world part that needs a backend is the sign to move the world into its own crate below the session, which Cargo then holds, not to lengthen that list. |

Verified surface, as of `main`:

| Where | How |
| --- | --- |
| macOS, Apple M4, Metal | every round |
| WebAssembly under WASI (`wasmtime`), the interpreter | the example replays, `serve` on standard input |
| A page: Entelechy as WebAssembly in headless Chrome, on its JIT (WebAssembly) and on WebGPU, files handed in | the web bridge's test, from `cargo test --workspace` (needs `wasm-bindgen` at Cargo.lock's version) |
| 32-bit WebAssembly under node: the core's, the front end's, the interpreter's and the JIT's tests, the JIT's code WebAssembly | `crates/interface/entelechy/tests/wasm.rs`, from `cargo test --workspace` |
| Interpreter, JIT, GPU | `examples/*/tests/*.replay` on all three |
| 36 language features × 7 execution forms (the modes, and a map of several outputs), and 8 narrow encodings read and written | `crates/backends/conformance`, JIT and GPU against the interpreter |
| 8 features of `Double` and `Long` × 6 modes; 5 of `Long` alone in maps and reductions | the same: the JIT against the interpreter, the GPU for `Long` |
| Keyboard, pointer | by hand |
| Pad sticks, buttons, triggers | by replay (`stick`, `button` lines) |
| Random programs: the JIT bit for bit, the GPU's integers and booleans exactly | `crates/backends/{jit,gpu}/tests/fuzz.rs`, from `entelechy_conformance::random`; `ENTELECHY_FUZZ=<n>` runs more |
| Mutated documents and protocol requests never crash or hang the front end or a session | `crates/interface/entelechy/tests/hostile.rs` |
| A session's random life: a success advances by one, a failure changes nothing, the Host does not grow | `crates/interface/session/tests/properties.rs` |
| Blender 5.2, headless Chrome, Godot 4.6, C ABI | `cargo test --workspace`, each skipped when its tool is absent |
| A live Blender | its MCP add-on, `127.0.0.1:9876`, `{"type":"execute","code":…,"strict_json":false}` + `\0` |

## Next

No release is planned; `v0.0.2-rc.3` is the last tag.

Open:

1. A page on Windows (an RTX 4090) did not open a large scene that a
   page on the M4 opens; nothing is known of why until that machine's
   browser console is read.
2. A page's WebAssembly memory stays at its peak, about 256 MB for a
   large scene: some 67 MB are Fields kept both in memory and on the
   device because the processor reads them each frame, some 60 MB the
   allocator's gaps.

What a page does only because one browser fails otherwise, each to be
taken back once that browser no longer fails:

- A page's device declines prefix passes: Chrome's WebGPU on Apple GPUs
  loses the device running them over a count that changes frame to
  frame (`GpuBackend::declines`).
- The GPU runs a long builder holding a loop counted at run time as a
  loop carrying its array: Chrome's WebGPU, its robustness on, computes
  it wrong as a builder's loop (`entelechy_ir::carried_builders`, applied
  in the GPU's `prepare_kernel`).

Measured on a 53-Kernel game (21 documents, 318 functions): `check`
0.3 s, a first frame 1.8 s on the JIT, an idle sync 0.23 ms. Left by
choice, each measured as a small gain for new complexity: a thread pool
for the JIT (a broad gain, but one scene slowed by thread scheduling),
preparing a document's Kernels in parallel (7%), one GPU submission a
frame (5 to 20%, every upload made an encoder copy). Two costs stay
super-linear until a real project meets them: a `use` chain hundreds of
libraries deep, and a document of thousands of Kernels and structs.
