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

Every state has an owner, and every rule one. A rule found with a second owner
is left with one, and held there by the compiler or by a row in
`crates/interface/orla/tests/owners.rs`; see `docs/DECISIONS.md`, *A rule that
had two owners is held to one*. Prefer immutable values, explicit mutation,
explicit lifetime, explicit dependencies.

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

What an agent trial finds wanting is recorded, once fixed, as data in
`crates/interface/orla/tests/findings/`: the request, and what its answer
must hold. A fix follows the rule's existing owner, and changes every
place that consumes it, the bridges and documents included; the findings
already recorded hold it to not breaking what was fixed before.

## Comments and documents

A comment or a document states present facts: what a thing is, who owns
it, what it guarantees.

- History is git's; reasons are `docs/DECISIONS.md`'s. No "used to", "no
  longer", "before".
- State what is done and where, not what is not; a negative stays only
  where it is the boundary itself ("no capability, no access").
- No narrative, comparison, hedging, or dated observation.
- One fact, one place: a comment holds its code's contract, a document
  the contract across modules; neither restates the other.
- A test's comment states what it holds; the regression it guards is
  its commit's message.

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

Cargo never removes an artifact whose fingerprint changed: every version
bump, and every `-p` set that unifies features differently, leaves a copy
of each binary with its debug info. After a release, `cargo clean` the
working target directory and remove the directories the release was built
in.

## Verification

A commit on a branch is verified by the tests of the crates it touches,
formatted and linted, and by its own regression test, seen to fail
without the change. A branch is verified by all of this once, before it
merges, and compared [against main](#against-main); a failure is fixed on the branch and the whole run repeated. Run
it in the background: each step is timed into `target/verify.log`, which
`tail -f` shows as it goes, with its output in `target/verify-<step>.out`.
Steps that need nothing of one another run at once; the first to fail stops
the run after the steps beside it, its output in the log.

```sh
log=target/verify.log; : > "$log"
step() {
  local name=$1; shift; local start=$SECONDS
  if "$@" > "target/verify-$name.out" 2>&1; then
    printf '%-14s %4d s\n' "$name" $((SECONDS - start)) >> "$log"
  else
    printf '%-14s %4d s  FAILED\n' "$name" $((SECONDS - start)) >> "$log"
    tail -40 "target/verify-$name.out" >> "$log"; return 1
  fi
}
# Steps that need nothing of one another, at once, each a function of its
# name; the run goes on when all of them passed.
together() {
  for name in "$@"; do step "$name" "$name" & done
  wait
  ! grep -q FAILED "$log"
}
# The bridges wait on the programs they drive, so they run beside the rest.
tests() { cargo test --workspace -- --skip bridge --skip blender; }
bridges() { cargo test -p orla --test bridges; }
# Replays are independent: several at once, each backend in turn; WASI has
# no threads, so each replay there is one core, and eight run at once.
replays() {
  for b in gpu cpu jit; do
    printf '%s\n' examples/*/tests/*.replay |
      xargs -P 4 -n 1 ./target/release/orla replay --backend "$b" || return 1
  done
  # The device runs every Kernel itself too, nothing left to the processor
  # by its size.
  printf '%s\n' examples/*/tests/*.replay |
    xargs -P 4 -n 1 ./target/release/orla replay --backend gpu --small 0
}
wasi() {
  cargo build --release -p orla --target wasm32-wasip1 &&
  printf '%s\n' examples/*/tests/*.replay |
    xargs -P 8 -n 1 wasmtime run --dir . target/wasm32-wasip1/release/orla.wasm replay --backend cpu
}
# See Against main. Main's build is kept between runs, so an unchanged
# main builds in seconds.
against_main() {
  local out=target/against-main
  rm -rf "$out/main" "$out/branch" "$out/tree"; git worktree prune
  mkdir -p "$out/main" "$out/branch"
  git worktree add -f "$out/tree" main &&
  CARGO_TARGET_DIR="$out/target" cargo build --release -p orla --manifest-path "$out/tree/Cargo.toml" || return 1
  for scene in examples/*/scene.osn; do
    local example=$(basename "$(dirname "$scene")")
    for side in main branch; do
      local orla=./target/release/orla; [ "$side" = main ] && orla="$out/target/release/orla"
      local at="$out/$side/$example"
      "$orla" check "$scene" > "$at.check" 2>&1
      "$orla" graph "$scene" > "$at.graph" 2>&1
      "$orla" run "$scene" --backend cpu --window 64x48 --out "$at.ppm" > "$at.run" 2>&1
      [ -f "$at.ppm" ] && md5 -q "$at.ppm" > "$at.picture" && rm "$at.ppm"
    done
  done
  git worktree remove --force "$out/tree"
  diff -r -I 'evaluated in' "$out/main" "$out/branch"
}
step fmt cargo fmt --all &&
step clippy cargo clippy --workspace --all-targets -- -D warnings &&
step clippy-wasm cargo clippy -p orla-jit -p orla-ffi --features orla-jit/wasm-host --target wasm32-unknown-unknown -- -D warnings &&
step clippy-page cargo clippy -p orla-gpu -p orla-session --target wasm32-unknown-unknown -- -D warnings &&
step test-build cargo test --workspace --no-run &&
together tests bridges &&
step release cargo build --release &&
together replays wasi against_main &&
echo passed >> "$log"
```

A step far slower than it was before is a fault to find, not a wait: stop
the run and look at that step alone. A debug build that slows down every
unit alike has most often been `target/debug/deps` filling with files; count
them (`ls -f target/debug/deps | wc -l`) and compare with a build in a fresh
`CARGO_TARGET_DIR` before blaming a setting. Debug info is packed into a
`.dSYM` (`split-debuginfo` in `Cargo.toml`) so that directory stays small.

A replay under `examples/*/tests/` is a test of what its example promises; a
piece made with the example is not one, and is not kept there.

### Against main

Tests hold what a change meant to do, not what it did beside it. Before a
branch merges, every example is also run by `main`'s build and by the
branch's, and what each says is compared: its `check`, its `graph` (the
plan), and its picture. Every difference is one the branch means, and its
commit says so; any other is a fault to find before the merge.
The `against_main` step of the verification does it, in
`target/against-main`: `main` checked out beside the branch and built into a
target directory kept between runs, and every difference printed.

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
