# Orla Handoff

## Context

The conversation with the user is Chinese; source, comments, documents and
commit messages are English.

`main` is the line. A round of work is one short branch off it (`feat/…`,
`fix/…`, `docs/…`), verified, then fast-forwarded back when the user says so.
There is no remote: nothing is pushed, and the CI workflow has never run.
Commit as the repository's configured identity.

Before a commit, in this order: `cargo fmt --all`, `cargo clippy --workspace
--all-targets -- -D warnings`, `cargo test --workspace`, then `cargo build
--release` and every replay on the three backends:

```sh
for f in examples/*/tests/*.replay; do
  for b in gpu cpu jit; do ./target/release/orla replay "$f" --backend "$b"; done
done
```

The interpreter is the reference: where the three disagree, it is right. A test
that writes files takes a scratch directory of its own.

The previous implementation at `../bak/orla` is a read-only reference, not a
source to copy from.

The user's Blender runs an MCP add-on on `127.0.0.1:9876` that executes Python
sent as `{"type":"execute","code":…,"strict_json":false}` terminated by `\0`,
answering the same way. That is how the Blender bridge was driven against a
live Blender; the headless test in `bridges/blender/README.md` needs none of it.

## Next

A play round on what the prefix mode just made cheap: a scene that orders what
it draws — a counting sort over a few hundred thousand things, the histogram a
scatter with `sum`, the ranks `prefix with sum`, the placement a scatter — and
whatever writing it turns up, fixed.

Done when the example lives in `examples/<name>/` with its own `kernels/`,
scene, README and replay; the replay passes on the interpreter, the JIT and the
GPU, pinning counts with no tolerance where the work is integer; anything it
turns up in the engine is fixed with a test of its own; and the verification
above is clean.

## Open

Unverified except by hand, and never verified by hand: the Player's right and
middle mouse buttons and its wheel, gamepad hardware (none has been connected —
a replay drives a pad instead), and a live window resize by dragging a corner.
The Godot bridge's window opens without script errors but has not been clicked
through; the Blender bridge has run on Blender 5.2 only.

Orla has only ever been built and run on this machine, macOS on an Apple M4. The
Linux build and every device path that is not Metal are unverified, and several
known workarounds are Metal's (timestamp queries that resolve to zeros, narrow
signed division, a loop counter whose generated name spelled `i32`).
