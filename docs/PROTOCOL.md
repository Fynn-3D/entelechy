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
compose   compose with expect = <version>, text or edits (checks, then rewrites the scene document and syncs), param
observe   evaluate, read (rgba32f or ppm, base64)
repair    the same operations again; retire; reset
```

`compose` checks the text before it writes it: a text that does not lower,
names a source no node or input provides, wires nodes round in a cycle,
or defines a Kernel that does not compile, is answered with a
`scene_rejected` event as a sync would answer it, and the document on disk
and the scene running stay as they were. `document` answers the scene's
`version`, a digest of its text; `compose` answers the version it wrote,
and with `expect` set to a version it writes only if the document is
still that one, otherwise it refuses with the version it is at.

The new text is given whole as `text`, or as `edits` of the document on
disk, made in order, each on what the one before left. An edit names one
statement by its kind and name (`node`, `output`, `param`, `input`,
`kernel`, `struct`, `type`, `fn`, or `use` with its path) or, with
`member`, one member of a node's block (`{"node": "castle", "member":
"size"}`), and gives its `text`: the statement or member whole,
declaring what the edit names, or `null` to remove it. What is written
is replaced, what is not is added after the last of its kind (a member
laid out as its block is), and only those bytes change: every other
statement, comment and blank line stays as written. A statement removed
takes the comment lines right above it. An edit that cannot be made (a
text declaring something else, or more than one thing; nothing there to
remove) refuses the whole `compose`, naming the edit, and writes
nothing; the result is then checked as any text is.

```json
{"op": "compose", "expect": "<version>", "edits": [
  {"node": "castle", "text": "node castle: castle.build@1 { base <- ground.result; }"},
  {"node": "view", "member": "voxels", "text": "voxels <- castle.result;"},
  {"node": "old", "text": null}
]}
```

A served scene is composed by an agent and read back through `read`, so
its inputs read files only beside the documents that declare them (the
scene and the graphs it places) and in directories the person started the
server with `--allow file.read=<dir>` for; a file elsewhere is a
`rejected` event naming it. Writes take `--allow file.write=<dir>` in the
same way. Neither is ever granted over the wire.

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
text frames, so a browser is a client too: see `bridges/web`. A GET that
asks for no upgrade is told, in a 426, what it reached; any other HTTP
request is answered with a 400 and closed, so what a page sends in its
body is never read as requests. Any page the person visits can reach
this machine, so the upgrade is refused (403) to a page elsewhere: a
page served from this machine (`localhost`, `127.0.0.1`, `[::1]`), one
opened from a file (`null`), and a client that is no browser connect;
another origin only where the server was started with
`--allow-origin <origin>`, `https://host:port` as a browser writes it.

`status` answers with the session's backend, scene, watched files,
counts, and `versions`, the ABI, language, IR and package format
versions each on their own, as `orla version` prints them, and
`grants`, the capabilities a Kernel may be given: every one, unless the
person started the server with `--grant` naming fewer. A Kernel that
requires one not granted is refused when it is prepared, with the ones
it lacks, and whatever ran before goes on running.

The world has a position, `revision` and `frame` in `status`: the
revision moves when the world changes other than by running (a Kernel
published or retired, the scene read again, a param set, state written
or reset), the frame counts the evaluations committed since. `history`
(`since` a revision) says what made each revision after it, and where
it came from: a document on disk, or the protocol and its channel
(`stdin`, `socket 3`, `websocket 4`, `orla_request` for a program or a
page that holds the session in its own process), a fact about how it arrived, not
who sent it (an edit found on disk while a request is answered is its
document's; the scene a `compose` or a `param` loads is the request's); and the attempts refused since, with why, which move
nothing. `kernel` with a `generation` reads the source a Kernel had then,
while the session remembers the change that published it, and
`document` with a `version` the scene document a `scene` change loaded:
composing it again, against the version that stands, rolls the
structure back as publishing an old source rolls a law back. What it
remembers is bounded; the session remembers changes, never whole worlds.

`snapshot` takes the world whole for its caller to keep: the revision and
frame, every Kernel with its generation and source, and every port's
committed state (`layout`, `shape`, `extent`, `data` as base64 words),
under `"snapshot": 1`, the format's version. Only committed state is
taken: after an evaluation that failed, `read` shows the output it
computed, the snapshot the state that stands. With `to`, the snapshot is
written to that file instead, where the server was started with `--allow
file.write`.

`restore` brings a snapshot's world back, given as `snapshot` gave it
or read `from` its file (where the session may read). `laws` has no
default: `"current"` puts the snapshot's state under the Kernels that run
now, migrated where a layout changed, restarted where nothing carries
it; `"snapshot"` publishes the snapshot's Kernels back first, all
together, where their sources differ. A state port the snapshot does not
name starts again from its initial source. The world is then proved by
one frame, under the context given as `evaluate` takes it, which commits
nothing: when it runs, the world is the snapshot's, and the next
`evaluate` is the frame after it; when it fails, the state, the laws (their generations too) and the
revision are what they were, and the attempt is in `history`. Runs off the frame
are let go either way. `structure` (`"current"` when absent) brings the
snapshot's scene document back too, with `"snapshot"`: checked as
`compose` checks it and written first, its own Kernels in place before
the trial frame, and put back with everything else when the world does
not run.

`kernels` lists the backend's native operations alongside published
Kernels, at generation 0 (a manifest's `memories` names the ports that
are an input and an output at once, and `budget` the loop iterations one
element may run; a port's `type` is written as the
Kernel writes it, `Grid<Cell>` and not the struct's number, though what
binds to it is still decided by its layout and never by that name); `kernel` (named in the `kernel` field, since a
request's own `id` is the envelope's) returns their contract where a
Kernel would show its source, and `publish` under a native id is refused.
The source `kernel` returns is what `publish` takes, for a node's inline
Kernel too: the id the node gave it is written in and the node's `domain`
left out, so an agent may fetch, edit, and republish any Kernel it sees.

The same operations are reachable inside a process that loads Orla as a
library: `orla_request` in [`orla.h`](../crates/interface/ffi/include/orla.h) takes
one line and returns its answer, beside the C functions that evaluate
and read Fields in place.

Operations, fields, and results are listed in
[the protocol module](../crates/interface/protocol/src/lib.rs), which is the
authoritative definition; [the end-to-end test](../crates/interface/orla/tests/protocol.rs)
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
the rules `run --save` follows, into a directory the session was started
with `--allow file.write` for and nowhere else; a path outside one is
refused before any frame runs. The answer says `frames` (how many
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
`median_ms`, `nodes` (each `{node, ms, backend, routed, uploaded,
downloaded}`, the costliest first, as `placements` below) and
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
unknown name is an error listing similar ones in `similar`. It is what
the compiler accepts, generated from its tables. An operation that is
not one is an error listing every one in `operations`. An evaluation
that fails names its `node` when the failure is one node's, and a state
migration that fails names its `port` and `migration` too, so an agent
reads that node's state and sees the layout the migration must take;
in `advance`, `frame` says which frame it was.

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
`placements`, `{node, backend, routed, uploaded, downloaded}` for the
same nodes: the backend that ran it, why the main one did not when it
did not (a scan on a GPU session, a `Double`), and the bytes its run
brought to a device from Host memory and back, which is what leaving
the device costs;
`migrated`, `{node, port, kernel}` for every `<~` port whose state was
carried into a new layout by its `migrate` Kernel before the nodes ran;
and `reset`, `{node, port, reason}` for every such port that had no
migration and restarted from its initial source; `pending`, the `async`
nodes whose run off the frame was going on, their readers taking the
last output; and `landed`, those whose run landed before this
evaluation's nodes ran. A `graph` answer shows a `<~` binding's
migration as `migrate`, a node's `async`, and where each node runs, as
`backend` and `routed` (why it is not the main backend, or null).

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

`reset` forgets committed state, so the next evaluation's `<~` readers
start again from their initial sources. With a `node` (and a `port`,
`result` when none is named) it forgets that port's alone: the rest of
the world keeps its state, and the port must be state as it must for
`write`. A run off the frame that began from the state forgotten, or
from what was computed from it, is let go rather than left to land it
again; other runs go on. History records which: `{kind: "reset"}`, with
the `node` and `port` for one.

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
version. `sync` reports `routed`, `{node, backend, why}`, when a node's
placement off the main backend begins or changes, once. A Kernel nothing runs yet, and `publish` itself, prepare inline so
the reply is final.
