# Creation protocol

`orla serve <scene.osn>` drives a live session over standard input and
output: one JSON object per line in, one per line out. Any agent in any
language can create through it, with the same authority rules as the CLI
and the Player: Kernels are prepared before they are published, publication
is atomic and checked against an expected generation, a rejected change
leaves the running version alive, and state commits only when a whole
evaluation succeeds.

```text
{"id": 1, "op": "kernels"}
{"id": 1, "ok": {"kernels": [{"id": "life.step@1", "generation": 3, "requires": [...], "inputs": [...], "params": [...], "outputs": [...]}]}}
{"id": 2, "op": "prepare", "source": "kernel x.y@1 { ... }"}
{"id": 2, "error": {"message": "`nope` is not defined", "line": 4, "column": 20}}
```

The loop an agent runs:

```text
inspect   status, kernels, kernel, graph, document
propose   prepare (diagnostics, nothing published)
publish   publish with expect = "absent" | "any" | <generation>
compose   compose (rewrites the scene document and syncs), param
observe   evaluate, read (rgba32f or ppm, base64)
repair    the same operations again; retire; reset
```

The same protocol runs inside the Player with `orla-player --serve
<scene.osn>`: requests arrive on stdin while the window runs, an `evaluate`
answers with the next window frame under the person's real pointer and keys,
and `read` returns the frame the window shows. An agent and a person then
work on the same picture.

`kernels` lists the backend's native operations alongside published
Kernels, at generation 0; `kernel` returns their contract where a Kernel
would show its source, and `publish` under a native id is refused.

Operations, fields, and results are listed in
[the protocol module](../crates/cli/src/protocol.rs), which is the
authoritative definition; [the end-to-end test](../crates/cli/tests/protocol.rs)
drives the binary as a client. Values cross as JSON numbers (integers become
`Int`), booleans, and arrays of two, three, or four numbers. Kernels published
through the protocol are not owned by any document: a document that later
defines the same id republishes it from the file.

`evaluate` takes `time`, `rate` (the sound device's clock, for Kernels
that read it; zero when absent), `pointer`, `down`, and `keys`.

A saved edit to a Kernel that is already running is prepared on a worker
thread: `sync` reports `preparing` at once and `published` (or `rejected`)
on a later `sync`, and every `evaluate` in between runs the previous
version. A Kernel nothing runs yet, and `publish` itself, prepare inline so
the reply is final.
