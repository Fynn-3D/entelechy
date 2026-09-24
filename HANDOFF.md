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

Carry a rig's travel in the scene. A joint a Kernel writes is done:
`skin.aim@1` turns any joint onto a point over what the clip says, and
`examples/reach` holds a shoulder computed and an elbow authored. What a
game needs next is root motion: a clip whose root travels, the travel
taken out of the pose each frame and carried by the scene's own state
(a `<~` feedback of where the rig stands), so a walk moves the body
through the world instead of dragging it back when the clip loops.

Done when:

- the `skin` package has a Kernel that splits a pose's root travel from
  the pose, and an example carries it through feedback, with its
  `kernels/`, scene, README and replay;
- the replay passes on the interpreter, the JIT and the GPU, pinning
  where the rig stands after the clip has looped at least once;
- anything it turns up in the engine or in the `skin` package is fixed
  with a test of its own;
- the verification above is clean.
