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
`crates/interface/entelechy/tests/owners.rs`; see `docs/DECISIONS.md`, *A rule that
had two owners is held to one*. Prefer immutable values, explicit mutation,
explicit lifetime, explicit dependencies.

Do not silently recover from invalid runtime state. Return meaningful errors.
A failed hot reload keeps the previous valid Kernel alive.

These are stable boundaries; change them intentionally and test compatibility:

```text
Capability ABI
Entelechy IR
KernelID
Serialized Graph Format
```

What other programs and written sources build against is held to its
implementation by a test, and grows without breaking what came before;
see `docs/DECISIONS.md`, *A public contract grows; it breaks only with its
version*. The language and the scene syntax grow only by the owner's
word (`docs/DECISIONS.md`, *The language grows only by the owner's word*);
friction found while writing is met by a function, a package or an
example. When the owner has agreed to the growth, record it with
`ENTELECHY_BLESS=1 cargo test -p entelechy-compiler --test surface`; when the scene
syntax grows, add a document to `crates/frontend/scene/tests/surface/`
and record it with `ENTELECHY_BLESS=1 cargo test -p entelechy-scene --test surface`.

What an agent trial finds wanting is recorded, once fixed, as data in
`crates/interface/entelechy/tests/findings/`: the request, and what its answer
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
GitHub repository `Fynn-3D/entelechy`; its CI workflow runs when started by hand,
until a release spans platforms. Commit as the repository's configured
identity. The conversation may be in any language; everything written into
the repository is English.

Resolve a conflict by understanding both intended behaviors, then verify the
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
cargo xtask verify
```

[`xtask`](crates/interface/xtask/src/main.rs) runs, in this order:

- `fmt`, then `packages` (`cargo xtask packages`: every package named,
  laid out and layered as [packages](packages/README.md) says, and every
  Kernel it declares reached by its own tests), then `clippy` over the
  workspace with every target, then
  `clippy-wasm` (the JIT writing WebAssembly and the FFI) and
  `clippy-page` (the device and the session in a page), warnings denied;
- `test-build`, the workspace's tests built;
- `tests` and `bridges` at once: the workspace's tests, and the bridges,
  which wait on the programs they drive;
- `release`, the release build;
- `replays`, `wasi` and `against_main` at once. `replays` runs every
  example's and every package's replays by the release build, four at a time, on each
  backend in turn; then on the device with nothing left to the processor
  by its size (`--small 0`), with nothing fused (`--no-fusion`), and
  rounded apart (`--round-apart`; a replay whose checks hold two forms
  alike says so and is left out); then checks every package used in one
  scene beside every other, as a scene reaches them all in one namespace.
  `wasi` builds for WASI and holds it to the system interface: a
  session served on standard input, every answer `ok`, and the replays
  that read and write files, on the processor. Under WASI there is no
  device and no code made at run time, so the pictures are the other
  backends' to check. `against_main` is [below](#against-main).

A branch that changes nothing the Rust build reads since it left `main`
(`crates/`, `bridges/`, `.cargo/`, the workspace's manifests) leaves out
`fmt`, the clippy steps, `test-build`, `tests`, `bridges` and `wasi`, and
says so in the log: their answers are `main`'s. `cargo xtask verify
--all` runs every step; on `main` every step runs.

The log begins with the commit it verifies and ends `passed`.

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
tests establish behavior, the other documents hold contracts — the handoff
holds only what none of them can supply. Keep failures or environment
constraints there only while the next session needs them.

Before ending a session, inspect the diff, run the checks, and replace `Next`.
