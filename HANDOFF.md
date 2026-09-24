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

Let the raster package draw a large triangle across many invocations.
Whether the CPU and the JIT keep a native rasterizer is decided (they
do; `docs/DECISIONS.md` has the measurements), and the package now has
the fast path's cutout too. What the measurements turned up is the
package's own weakness: `raster.cover@1` gives each triangle one
invocation, which walks every pixel of its box, so a mesh of few large
triangles — the model example — draws 9.6 times as slow on the GPU as
the device's rasterizer, where a mesh of small ones is within 1.1 times.

Done when:

- a triangle whose box is large is covered by many invocations, each a
  tile of its box, by Kernels in the package and nothing in the engine;
- the model example's frame on the GPU with the package is within a
  small factor of the device's rasterizer, measured and written where
  the decision lives, and the torus and culled scenes lose nothing;
- `edges.replay`, `cutout.replay` and the session test that holds the
  package to the fast path to the bit still pass, unchanged;
- the verification above is clean.
