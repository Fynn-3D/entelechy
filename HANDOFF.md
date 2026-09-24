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

Give the ten-million-triangle frame room. The package draws
[`ten_million.osn`](examples/mesh/tests/ten_million.osn) on the GPU in
19.7 to 20.0 ms (the device's rasterizer: 32), which meets the bound
with half a millisecond to spare, and the frame is near what its memory
traffic allows (the rasterizer's decision in `docs/DECISIONS.md`). The
largest share is the mesh, not the rasterizer: `mesh.split_vertices`
makes three new vertices a face and shares no midpoint, so level 6 has
9.4 million vertices for 9.4 million triangles, and `project` writes
and the cover gathers twice what a shared mesh would need.

Done when:

- the torus at `level = 6` draws at 3840 by 2160 on the GPU in 17 ms a
  frame or less with the package, measured where the decision lives;
- the way there is Kernels and presets (a subdivision that shares its
  midpoints, or what measuring shows), and anything the engine needs for
  it is fixed with a test of its own;
- the mesh replays (`torus`, `edges`, `cutout`, `native`, `near`) still
  pass, or change only where the mesh itself is meant to, and the small
  scenes (torus, culled, model) lose nothing;
- the verification above is clean.
