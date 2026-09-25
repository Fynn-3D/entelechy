# Process

**Answers:** how code is written, verified, committed, and handed to the next
session.

## Code

Code should reveal the architecture. Not hide it.

Respect the layers. Do not bypass them for convenience.

```text
Host → ABI → IR → Kernel Runtime → Graph Runtime
```

One module, one responsibility. Avoid god objects, hidden globals, cross-layer
helpers, implicit ownership.

Every state has an owner. Prefer immutable values, explicit mutation, explicit
lifetime, explicit dependencies.

Do not silently recover from invalid runtime state. Return meaningful errors.
A failed hot reload keeps the previous valid Kernel alive.

These are stable boundaries; change them intentionally and test compatibility:

```text
Capability ABI
Orla IR
KernelID
Serialized Graph Format
```

What other programs and written sources build against is held to its
implementation by a test, and grows without breaking what came before;
see `docs/DECISIONS.md`, *A public contract grows; it breaks only with its
version*. When the language's surface grows, record it with
`ORLA_BLESS=1 cargo test -p orla-compiler --test surface`; when the scene
syntax grows, add a document to `crates/frontend/scene/tests/surface/`
and record it with `ORLA_BLESS=1 cargo test -p orla-scene --test surface`.

Prefer clear, small, boring, testable over clever, abstract, magical,
prematurely generic.

Formatting and basic style belong to the formatter, the linter, compiler
warnings, the tests. Do not spend attention on what machines enforce.

> **Formatting is automatic.
> Architecture is deliberate.**

## Git

`main` is the line. Work on a focused `feat/*`, `fix/*`, `refactor/*`,
`docs/*`, `test/*` or `chore/*` branch, one round of work each.

Before choosing a base, inspect the working tree, recent commits, and remotes.
Preserve user changes; never stash or discard unrelated work to clean the tree.
Work that depends on unmerged commits continues from that branch, not from an
older `main`.

Inspect the diff and run the checks that fit it: [Verification](#verification)
for a change to the code, links and references for a change to the documents.
Stage only intended files. Write small logical commits in English (`feat: add kernel
registry`, `docs: clarify external agent boundaries`). Record what was actually
verified, limits included.

Local-commit authorization is not authorization to push, merge, or release.
Pushing and opening a PR need a configured remote and explicit permission;
merging back to `main` happens on the user's word, a fast-forward. Never
force-push or rewrite history without being asked. `origin` is the private
GitHub repository `Fynn-3D/orla`; its CI workflow runs when started by hand,
until a release spans platforms. Commit as the repository's configured
identity. The conversation may be in any language; everything written into
the repository is English.

Resolve a conflict by understanding both intended behaviours, then verify the
combined result. Never take one side blindly.

Releases use semantic versioning, are tagged within release authorization, and
document intentional compatibility changes. A commit is not a release.

## Verification

A change to the code is verified by all of this before it is committed:

```sh
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p orla-jit -p orla-ffi --features orla-jit/wasm-host --target wasm32-unknown-unknown -- -D warnings
cargo clippy -p orla-gpu -p orla-session --target wasm32-unknown-unknown -- -D warnings
cargo test --workspace
cargo build --release
# Replays are independent: several at once, each backend in turn. A
# failure prints why and makes xargs exit non-zero.
for b in gpu cpu jit; do
  printf '%s\n' examples/*/tests/*.replay | xargs -P 4 -n 1 ./target/release/orla replay --backend "$b"
done
cargo build --release -p orla --target wasm32-wasip1
# WASI has no threads, so each replay is one core: eight at once.
printf '%s\n' examples/*/tests/*.replay |
  xargs -P 8 -n 1 wasmtime run --dir . target/wasm32-wasip1/release/orla.wasm replay --backend cpu
```

Run it in the background with each step timed, so its progress can be
watched; a step far slower than it was before is a fault to find, not a
wait. A replay under `examples/*/tests/` is a test of what its example
promises; a piece made with the example is not one, and is not kept there.

## Handoff

Repository state is authoritative. At session start follow the reading order in
`AGENTS.md`, inspect Git state, and verify the code `HANDOFF.md → Next` names.

`HANDOFF.md` carries two things and nothing else:

- **Context:** the facts of this repository an agent must hold to act, and the
  surface already verified.
- **Next:** one executable action and its acceptance criteria.

How the work is done, verified and committed is this document's, not the
handoff's.

Replace what is stale; never append a diary. Git records history, code and
tests establish behaviour, the other documents hold contracts — the handoff
holds only what none of them can supply. Keep failures or environment
constraints there only while the next session needs them.

Before ending a session, inspect the diff, run the checks, and replace `Next`.
