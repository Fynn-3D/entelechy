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

Inspect the diff and run the checks that fit it: `HANDOFF.md → Verification`
for a change to the code, links and references for a change to the documents.
Stage only intended files. Write small logical commits in English (`feat: add kernel
registry`, `docs: clarify external agent boundaries`). Record what was actually
verified, limits included.

Local-commit authorization is not authorization to push, merge, or release.
Pushing and opening a PR need a configured remote and explicit permission;
merging back to `main` happens on the user's word. Never force-push or rewrite
history without being asked.

Resolve a conflict by understanding both intended behaviours, then verify the
combined result. Never take one side blindly.

Releases use semantic versioning, are tagged within release authorization, and
document intentional compatibility changes. A commit is not a release.

## Handoff

Repository state is authoritative. At session start follow the reading order in
`AGENTS.md`, inspect Git state, and verify the code `HANDOFF.md → Next` names.

`HANDOFF.md` carries three things and nothing else:

- **Context:** the facts of this repository an agent must hold to act, and the
  surface already verified.
- **Verification:** the checks a commit needs here.
- **Next:** one executable action and its acceptance criteria.

Replace what is stale; never append a diary. Git records history, code and
tests establish behaviour, the other documents hold contracts — the handoff
holds only what none of them can supply. Keep failures or environment
constraints there only while the next session needs them.

Before ending a session, inspect the diff, run the checks, and replace `Next`.
