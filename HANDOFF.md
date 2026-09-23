# Handoff

**Answers:** what this repository holds that nothing else records, and the next action.

## Context

| Fact | What follows from it |
| --- | --- |
| Conversation is Chinese; source, comments, documents and commits are English. | Write artifacts in English whatever the request was written in. |
| `main` is the line; a round of work is one branch, fast-forwarded back on the user's word. | Branch first, ask once at the end, never merge unasked. |
| No remote is configured. | Commits are local; the CI workflow describes intent, not a run. |
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

Drive a rig rather than play one. The ABI changes the user asked for
are done (several outputs, narrow storage, parameters of any value
type, `Double` and `Long`); robust geometry predicates over doubles
are a later round, if the user wants them. `examples/skin`
blends two clips by where the pointer is and `examples/crowd` poses
sixty-four out of one rig; what a game needs next is a joint a Kernel
writes — a head that follows a point, a foot held on the ground —
between the sample and the palette, and root motion: a clip whose root
travels, the travel taken out of the pose and carried by the scene's own
state.

Done when:

- an example holds a rig whose pose is partly authored and partly
  computed, with its `kernels/`, scene, README and replay;
- the replay passes on the interpreter, the JIT and the GPU, pinning the
  joint matrices at two moments;
- anything it turns up in the engine or in the `skin` package is fixed
  with a test of its own;
- the verification above is clean.
