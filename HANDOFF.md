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
| 35 language features × 7 execution forms (the modes, and a map of several outputs), and 8 narrow encodings read and written | `crates/matrix`, JIT and GPU against the interpreter |
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

Parameter types, the third of three changes to the ABI the user asked
for (several outputs and narrow storage are done; f64 and i64 are
queued after, on the user's word; the rig work after that). A parameter
is a `Float`, `Int`, `Bool` or `Vec2`–`Vec4` today, so a camera's
`Mat4`, a cell's `IVec2` or a light's struct has to be rebuilt inside the
Kernel or passed through a Field. Let a parameter be any value type a
Kernel can hold — matrices, integer vectors, structs, fixed arrays —
carried as the Kernel's words like any other value.

Done when:

- `param view: Mat4 = mat4(1.0);`, an `IVec2`, a struct and an array
  parameter compile, with their defaults, and a graph sets them;
- the protocol, the C ABI and replay's `param` line take them, with
  a clear refusal where a value does not fit the type;
- the interpreter, the JIT and the GPU agree on them, held by rows of
  the feature matrix;
- an example uses one where it was rebuilding it before, with a replay
  on all three backends;
- the verification above is clean.
