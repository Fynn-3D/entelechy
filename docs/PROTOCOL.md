# Creation protocol

**Answers:** what an agent sends over the wire, and what comes back.

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

`prepare` and `publish` take a document holding one Kernel: the Kernel
sees the declarations the source makes or `use`s, and the scene's shared
ones after those, so a Kernel copied from a library file republishes as
it is, structs and all.

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

With `--listen <host:port>` (`127.0.0.1:7788`, or port 0 for any free
port, named on stderr as `listening on ...`), `orla serve` and the Player
also accept connections on a socket, the same lines in and out, one
response per request in order; every connection is served by the one
session, a `quit` on a socket ends only that connection, and a `quit` on
stdin ends the process. This is how another program, Blender say, talks
to a running window: see `bridges/blender`. A connection that opens with
an HTTP request is upgraded to a WebSocket and carries the same lines as
text frames, so a browser is a client too: see `bridges/web`. Anything
else opening with HTTP is told, in a 426, what it reached.

`status` answers with the session's backend, scene, watched files,
counts, and `versions`, the ABI, language, IR and package format
versions each on their own, as `orla version` prints them.

`kernels` lists the backend's native operations alongside published
Kernels, at generation 0 (a manifest's `memories` names the ports that
are an input and an output at once; a port's `type` is written as the
Kernel writes it, `Grid<Cell>` and not the struct's number, though what
binds to it is still decided by its layout and never by that name); `kernel` (named in the `kernel` field, since a
request's own `id` is the envelope's) returns their contract where a
Kernel would show its source, and `publish` under a native id is refused.
The source `kernel` returns is what `publish` takes, for a node's inline
Kernel too: the id the node gave it is written in and the node's `domain`
left out, so an agent may fetch, edit, and republish any Kernel it sees.

The same operations are reachable inside a process that loads Orla as a
library: `orla_request` in [`orla.h`](../crates/ffi/include/orla.h) takes
one line and returns its answer, beside the C functions that evaluate
and read Fields in place.

Operations, fields, and results are listed in
[the protocol module](../crates/session/src/protocol.rs), which is the
authoritative definition; [the end-to-end test](../crates/cli/tests/protocol.rs)
drives the binary as a client. Values cross as JSON numbers, booleans,
and arrays, as the paragraph on values below says. Kernels published
through the protocol are not owned by any document: a document that later
defines the same id republishes it from the file.

`evaluate` takes `time`, `rate` (the sound device's clock, for Kernels
that read it; zero when absent), `pointer`, `down`, `buttons` (the pointer
buttons held, by number: 0 left, 1 right, 2 middle), `wheel` (what scrolled
since the last evaluation), `keys`, and the window as `window` (`[w, h]`
in pixels) and `scale` (its scale factor, one when absent); an evaluation
that names no window sees the session's, set by the `window` op (`size`
as `[w, h]`, `scale`; `size` null for none) or `orla serve --window
<w>x<h>[@<scale>]`, and none otherwise. Inside the Player the window is
the person's and these two are ignored. With `nodes`, it evaluates only
those nodes, what they read, the state (which must advance), and any
output never produced; the rest keeps its last value. An agent watching
a state row pays for the state, not the picture. With `settle: true`, it
first waits for every `async` node's run off the frame to land, so the
evaluation reads it.

`advance` runs many frames in one request, which is how a job that
spans them is asked for: `frames` (1 to 100000) at `fps` frames a
second of the clock (60 by default), from `time`, with the rest of
`evaluate`'s context held for every frame and an `async` node settled
after each. With `until`, `{target, below | above | equals}`, it stops
as soon as the value says to: the target is `node.port[index].field`
(a struct's field by name, a vector's component by its letter),
the spelling a replay writes, and the value is a Float, an Int or a
Bool. With `save`, an object of `node.port` to a file path (relative
paths beside the scene), it writes those outputs when the run stops, by
the rules `run --save` follows. The answer says `frames` (how many
ran), `stopped` (`"frames"` or `"until"`), `target` and `value` (what
the watched value last was), `saved`, `millis`, and `report`, the last
frame's. Only what `until` and `save` name has to be materialized, so a
training run does not draw the picture; with neither, the whole scene
runs. A network trained over a thousand frames and kept as a file is
one request, not a thousand.

`bench` measures what a change cost: `frames` frames (30 by default) at
`fps` (60), run `repeat` times (3), of which the least is reported, as
`orla bench` reports it from a terminal. `frames` times `repeat` is at
most 100000. The first frame of all is thrown away, since it compiles
and allocates; with `nodes` true, one more run times every node with
the device waited for after each, and a node placed several times is
one line. The answer says `backend`, `frames`, `repeat`, `best_ms`,
`median_ms`, `nodes` (each `{node, ms}`, the costliest first) and
`ran`, which is false when the scene does not change with the clock and
every node was reused after the first frame. The clock advances while
it measures, as `advance` advances it, so a measurement is also a run:
an agent that edits a Kernel can ask what the edit cost, and compare it
with what it asked before.

`help` answers with every builtin a Kernel may call (`builtins`, each
`{name, kind, signatures, doc, requires, entry_only}`) and every form
the language and the documents are written in (`forms`, each `{name,
kind, shape, doc}`: `loop`, `each`, `scatter`, `node`, `fill`,
`previous`, `async` and the rest), or, with `name`, whichever it is; an
unknown name is an error naming similar ones. It is what the compiler
accepts, generated from its tables.

`trace` runs one element of a node on the interpreter, over the Kernel
as its document wrote it (compiled again without the optimizer, so the
calls and the `let`s are the source's), and answers with
every binding of it: `node`, and `at` (`[x, y, z]`, an element of a map)
or `element` (a step of a scan), with the fields `evaluate` takes for the
context; the scene is evaluated first under that context, with the
node's sources pinned so their Fields exist, and the trace is of the
evaluation that would follow (its `<~` inputs the state just committed),
nothing being committed by the trace itself. The answer has `kernel`,
`at`, `bindings` and `results`, each a list of `{name, type, value, ran}`
with a struct as an object by field name and `ran` false for a `let` in
a branch the element did not take, and `calls`, each `{function,
bindings, results}` in the order the calls returned (`calls_truncated`
when more than 256 were made). An evaluation's report carries
`timings`, `{node, millis}` for every node that executed, in order;
`migrated`, `{node, port, kernel}` for every `<~` port whose state was
carried into a new layout by its `migrate` Kernel before the nodes ran;
and `reset`, `{node, port, reason}` for every such port that had no
migration and restarted from its initial source; `pending`, the `async`
nodes whose run off the frame was going on, their readers taking the
last output; and `landed`, those whose run landed before this
evaluation's nodes ran. A `graph` answer shows a `<~` binding's
migration as `migrate` and a node's `async`.

`read` with `format: "field"` returns any Field as the Host's file format
(what `run --save` writes and an `input` reads), and `write` with the same
format takes one back, checked against the port's layout and extent.
`read` with a `node` returns a Field that is not pixels as its words, and
`write` with a `node`, a `port`, and `data` (base64 words in the same
layout) replaces that port's committed state, so the next evaluation's
`<~` readers start from it. This is how a Field comes in from outside, a
saved file for instance; the port must be state, read by some `<~`
binding, and the count of words must match. A `read` after an evaluation
and a `write` before the next make a save and a load. The words are the
Kernel's, one per scalar (two for a `Double` or a `Long`, the low one
first), whatever the port keeps: a Field kept narrow
(`Grid<Vec4 as unorm8>`) is read widened and written narrowed by the
Host's rule, and only the `field` format carries its packed bytes.

A value in a request is JSON: a number (an integer is an `Int`, a `Long`
past 32 bits, and stands in for a float; a number with a fraction is
exact until the parameter it is given to says its width), a string of
digits for a `Long` past 2^53, which a JSON reader keeps exact where it
would round a number, `true` or `false`, or an array of values in
order, which the parameter it is given to takes in its own shape: a
vector's components, a matrix's columns (`[[3, 0, 0, 0], [0, 3, 0, 0],
[0, 0, 1, -1], [0, 0, 0, 0]]` for a `Mat4`), a struct's fields or an
array's elements. A value of another layout is refused with the
parameter's type; a parameter the scene does not declare keeps what it
was given. Replies spell values the same way (a `Double` as the number
that reads back as it, a `Long` past 2^53 as its digits), and `kernel`
and `graph` name each parameter's type.

A saved edit to a Kernel that is already running is prepared on a worker
thread: `sync` reports `preparing` at once and `published` (or `rejected`)
on a later `sync`, and every `evaluate` in between runs the previous
version. A Kernel nothing runs yet, and `publish` itself, prepare inline so
the reply is final.
