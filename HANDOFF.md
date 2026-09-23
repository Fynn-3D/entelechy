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
| 35 language features × 7 execution forms (the modes, and a map of several outputs) | `crates/matrix`, JIT and GPU against the interpreter |
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

Narrow storage, the second of three changes to the ABI the user asked
for (several outputs is done; parameter types that are matrices,
integer vectors and structs come third; the rig work after them). Every
scalar of a Field is a 32-bit word, so a byte read with `bytes` costs
four and a pixel sixteen. Give the schema narrow scalars — 8- and
16-bit integers, normalized bytes, half floats — stored packed in the
Host, on the device and in `.field` files, widened when a Kernel reads
them and narrowed when it writes, so Kernels still compute in `Int` and
`Float`. The spelling of a narrow element in a port type is the user's
call; ask before settling it.

Done when:

- a port may name a narrow element and a PNG or a `bytes` input lands
  packed;
- a Field's size in the Host, on the device, in a `.field` file and
  over the protocol is its packed size;
- how a value is narrowed (rounding, clamping) is one rule in
  `orla-rules`, and the interpreter, the JIT and the GPU agree on it bit
  for bit, held by rows of the feature matrix;
- an example and its replay pass on all three backends;
- the verification above is clean.
