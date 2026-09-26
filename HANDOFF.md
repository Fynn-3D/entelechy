# Handoff

**Answers:** what this repository holds that nothing else records, and the next action.

## Context

| Fact | What follows from it |
| --- | --- |
| The interpreter is the reference; the JIT matches it bit for bit, the GPU to rounding. | Where three answers differ, `--backend cpu` is the right one. |
| A new Kernel's cost on the GPU is its pipeline's compile: 40 to 120 ms for the voxel example's Kernels on Metal, nearly all of preparing it; one compiled before comes from the driver's cache in under a millisecond. | Natively it is paid off the frame (the preparation worker). A page has no thread: a Kernel changed while a page plays stops its frames for the compile. Data, params and structure (`write`, `param`, a `compose` that changes no Kernel) never compile. Scenes are built in the native Player; a page shows them. |
| Making a page's compile smooth needs the browser's `createComputePipelineAsync`, which wgpu does not expose; reaching it takes `wasm-bindgen` and `js-sys` in `orla-gpu` for the page, which AGENTS.md does not allow. | Not done by choice (2026-09-26). Taken up only if Kernels are to be changed live in a page, and then with that rule changed first. |
| The session holds the world (scene, ledger, sources, history, inputs, writes) and also opens backends; only its `lib.rs`, `session.rs` and `observe.rs` may name one (`crates/interface/orla/tests/layers.rs`). | A world part that needs a backend is the sign to move the world into its own crate below the session, which Cargo then holds, not to lengthen that list. |

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

v0.0.2-alpha.2 is on `feat/typed-protocol`: the protocol's operation
table types each field and the server checks every request against it;
each operation's request as JSON Schema; an MCP door (`orla serve
<scene> --mcp`) whose tools are the operations. Accepted by a fresh
agent that reached Orla only over MCP. It is merged, tagged and
published on the user's word; a release is cut once a whole batch is
done and accepted, never per fix.

Left from that acceptance, not taken up: a field's type is checked
before the operation's own hint about its meaning is given; `compose`
edits append without a blank line; `read` answers a large picture as
base64 in full.
