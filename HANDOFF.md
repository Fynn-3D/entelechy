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

`main` holds, since `v0.0.2-alpha.3`, scaling, hot-reload and runtime
performance work. A Kernel
lowers only the functions it calls and republishes only when its IR
changes; a changed library function republishes every Kernel that uses
it; an `if` or `match` on a literal compiles to the block it takes. A
fused chain is prepared once; a plan is reused, uncopied, while its
graph and inputs hold; a sub-graph placed many times is read once; a
document is parsed once per stamp and a `use` expanded again only when a
directory or manifest it read changed. The
JIT reads a grid or volume at a float address inline and blends linear
reads in place; the GPU runs a repeated map's times in one pass and fills
whole workgroups over rows. Two faults are fixed: an extent past 2^24
elements on the GPU, and an `async` Kernel run on a backend other than
the one it was prepared on. `IVec4` is new.

The language writes a float with an exponent (`1e30`, `2.5e-3`, `3e2d`)
and names Ints with enums (`type Mode = enum { Idle, Walk = 10 };`), a
scene writing a variant where it writes a number. The runtime holds to
its own rules on every path an agent or a file reaches: a JIT frame is
bounded (a module past it is left to the interpreter, or refused where
the JIT is alone) and a float address clamps in integers; every held
read is judged where its path lands, at the store; a value's type is
bounded in size and depth; a file decodes only what it holds, within
the Host's room; every placed node counts against the scene's bound; a
connection is read no further while 64 of its answers wait; a narrow
Field read through the C ABI stays until the next evaluate, request or
close; a file is written new at its working name.

It is released as `v0.0.2-rc.3`, a pre-release; `v0.0.2` follows once
the user accepts it. The release notes name what changed for a caller
since `v0.0.2-rc.2`:
- New: the noise package (integer hashes the same on every backend,
  value, gradient, fractal, ridged, Worley and curl noise) and the image
  package (colour spaces, blend modes and compositing, filters, tone,
  LUTs, histograms, signed distance, sampling, and the looks a scene
  places last); the `nebula` and `darkroom` examples.
- Changed: the `picture` and `pixelfx` packages are the image package,
  their Kernels `image.*` (`mip.first` is `image.mip_first`,
  `pixelfx.pixelate` is `image.pixelate`).
- The language: functions call one another in any order, a chain of
  calls back to one refused naming it; a function, a struct and a type
  are scoped by the document that declares them; a repeated node's
  name is its last time's wherever a port is named.
- Replays: `frame <from> to <last> every <step>`, `show`, and `--out` on
  `replay` and `render` naming a directory of frames or a `.wav`.
- A failing element is traced to where it is written; `pads()`; a PNG is
  compressed; a refusal of an unchanged scene is said once; a package
  outside a served scene's reach is refused naming `ORLA_PACKAGES`.

Measured on a 53-Kernel game (21 documents, 318 functions): `check`
0.3 s, a first frame 1.8 s on the JIT, an idle sync 0.23 ms. Left by
choice, each measured as a small gain for new complexity: a thread pool
for the JIT (a broad gain, but one scene slowed by thread scheduling),
preparing a document's Kernels in parallel (7%), one GPU submission a
frame (5 to 20%, every upload made an encoder copy). Two costs stay
super-linear until a real project meets them: a `use` chain hundreds of
libraries deep, and a document of thousands of Kernels and structs.
