# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. Local commits are
authorized; do not merge or push. Preserve both architectural bibles verbatim.

This implementation started on 2026-09-19 from the architecture documents alone.
The previous implementation lives outside the repository at `../bak/orla` as a
read-only reference; study its invariants, do not port it wholesale. Its typed
Field decision remains the binding direction for numerical resources when they
are introduced; the current Host exposes only images.

The workspace has no external dependencies. The language covers the Phase 1
constructs only: one `compute` block per Kernel, one Image output, helper
functions without recursion, scalar and vector parameters, no control flow.
`.osn` scenes bind nodes to `KernelId`s; the CLI session resolves sources from
the filesystem by polling modification stamps.

## Next

Open the creation surface to external agents: a process-level protocol over
stdin/stdout (JSON lines) on the `orla` binary that exposes the session
operations already present in `crates/cli/src/session.rs`: list Kernels and
manifests, inspect the graph, prepare a source and return diagnostics, publish
against an expected generation, evaluate, and export. Acceptance: a test in
`crates/cli/tests/` drives the binary as a client, receives structured
diagnostics for a broken Kernel, and observes that the running version
survived. Do not add product logic to the Host or the runtimes; the protocol is
an interface over existing contracts.

After that, Phase 2 begins with a GPU backend behind the same IR.

## Open

The watch loop polls every 200 ms and can observe a half-written file; the next
write is picked up because the stamp changes again. Exports are PPM only.
The example has no authored `tests/` directory yet: verification Kernels and
declarative replay, as AGENTS.md requires for demos, need a comparison primitive
that the language does not have. Add that before the second example.
