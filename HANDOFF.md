# Handoff

**Answers:** what this repository holds that nothing else records, and the next action.

## Context

| Fact | What follows from it |
| --- | --- |
| Conversation is Chinese; source, comments, documents and commits are English. | Write artifacts in English whatever the request was written in. |
| `main` is the line; a round of work is one branch, fast-forwarded back on the user's word. | Branch first, ask once at the end, never merge unasked. |
| `origin` is the private GitHub repository `Fynn-3D/orla`. | Push only on the user's word; the CI workflow runs when started by hand, until a release spans platforms. |
| The interpreter is the reference; the JIT matches it bit for bit, the GPU to rounding. | Where three answers differ, `--backend cpu` is the right one. |
| `../bak/orla` is the previous implementation, read-only. | Consult it; take nothing from it unchecked. |
| Commit as the repository's configured identity. | `git config user.name` is already what the user wants. |

Verified surface, as of `main`:

| Where | How |
| --- | --- |
| macOS, Apple M4, Metal | every round |
| Interpreter, JIT, GPU | `examples/*/tests/*.replay` on all three |
| 35 language features × 7 execution forms (the modes, and a map of several outputs), and 8 narrow encodings read and written | `crates/backends/conformance`, JIT and GPU against the interpreter |
| 8 features of `Double` and `Long` × 6 modes; 5 of `Long` alone in maps and reductions | the same: the JIT against the interpreter, the GPU for `Long` |
| Keyboard, pointer | by hand |
| Pad sticks, buttons, triggers | by replay (`stick`, `button` lines) |
| Blender 5.2, headless Chrome, Godot 4.6, C ABI | `cargo test --workspace`, each skipped when its tool is absent |
| A live Blender | its MCP add-on, `127.0.0.1:9876`, `{"type":"execute","code":…,"strict_json":false}` + `\0` |

## Verification before a commit

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release
for f in examples/*/tests/*.replay; do
  for b in gpu cpu jit; do ./target/release/orla replay "$f" --backend "$b"; done
done
```

## Next

Decide whether the CPU and the JIT still need a native rasterizer. The
rig is driven now (`skin.aim@1` and `examples/reach` for a joint a
Kernel writes; `skin.stride@1`, `skin.travel@1` and `examples/stride`
for root motion), and the raster package is within about 1.1 times the
device's rasterizer on the GPU and 1.4 times the Rust fast path on the
JIT. `native.rasterize@1` on the CPU and the JIT is the rasterizer's
rule written a second time, in Rust, held to the package by a test;
the GPU's is the hardware, which the typed-field architecture allows.

Done when:

- the package and the fast path are measured side by side on the JIT
  and the interpreter over the mesh, culled and model examples, small
  meshes and large;
- either the fast path is kept, with the measurements that justify it
  written where the decision lives, or it is retired on the CPU and the
  JIT and every scene that named it runs the package there, its replays
  unchanged;
- the verification above is clean.
