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

Rasterize ten million triangles at 4K within a frame. Large triangles
are handled (the `tiled` raster presets, opt-in); what a big scene hits
now is memory, not arithmetic. At 3840 by 2160, 9.4 million triangles
(the torus at `level = 6`) take 33 ms with the device's rasterizer and
about 47 with the package, and neither is mostly the coverage test:
`native.rasterize` writes a seven-word surface a pixel (232 MB at 4K),
`project` writes nine words a vertex, the package's cover reads three
whole vertices a triangle for the four words of position it needs, and
the shade reads the surface back. The package's keys are already a
visibility buffer — a depth and a triangle in eight bytes a pixel.

Done when:

- a scene draws the torus at `level = 6` at 3840 by 2160 on the GPU in
  20 ms a frame or less, measured and written where the decision lives,
  with the package (not only the device's rasterizer);
- the way there is Kernels and presets — shading from the keys by the
  triangle rather than from a materialized surface, a position stream
  the cover reads instead of whole vertices, or what measuring shows —
  and anything the engine needs for it is fixed with a test of its own;
- `edges.replay`, `cutout.replay` and the session test that holds the
  package to the fast path to the bit still pass, and the small scenes
  (torus, culled, model) lose nothing;
- the verification above is clean.
