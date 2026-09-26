# Creation protocol

**Answers:** what an agent sends over the wire, and what comes back.

`orla serve <scene.osn>` drives a live session over standard input and
output: one JSON object per line in, one per line out. Any agent in any
language creates through it under the same authority rules as the CLI
and the Player: Kernels are prepared before they are published,
publication is atomic and checked against an expected generation, a
rejected change leaves the running version alive, and state commits only
when a whole evaluation succeeds.

```text
{"id": 1, "op": "kernels"}
{"id": 1, "ok": {"kernels": [{"id": "life.step@1", "generation": 3, "requires": [...], "inputs": [...], "params": [...], "outputs": [...]}]}}
{"id": 2, "op": "prepare", "source": "kernel x.y@1 { ... }"}
{"id": 2, "error": {"message": "`nope` is not defined", "line": 4, "column": 20}}
```

Every operation, its fields and its answer are in
[Operations](#operations); an unknown operation is an error listing
every one in `operations`.

The loop an agent runs:

```text
inspect   status, kernels, kernel, graph, document
propose   prepare (diagnostics, nothing published)
publish   publish with expect = "absent" | "any" | <generation>
compose   compose with expect = <version>, text or edits (checks, then rewrites the scene document and syncs), param
observe   evaluate, read (rgba32f or ppm, base64)
repair    the same operations again; retire; reset
```

## Operations

A request is one JSON object a line, with an `op` and an optional `id`
echoed back. An answer is one line, `{"id", "ok": ...}` or `{"id",
"error": {"message", "line"?, "column"?}}`. An error carries what acts
on it as data: an unknown operation the `operations`, an unknown `help`
name the `similar` ones, a failed evaluation its `node` (a state
migration's also `port` and `migration`) and a run's `frame`. A field
marked `?` may be left out.

| op | fields | answer |
| --- | --- | --- |
| `status` |  | backend, scene, watched, kernels (native operations included), nodes, revision, frame, fusion, grants, versions (`orla`, `api`, `language`, `ir`, `package`) |
| `kernels` |  | a list of manifests, each with its generation, which `publish`'s `expect` names |
| `kernel` | `kernel`?: string, `generation`?: integer | manifest, generation, source, as `publish` takes it |
| `graph` |  | inputs, params, nodes, outputs |
| `document` | `version`?: string | path, text, and version, which `compose`'s `expect` names |
| `history` | `since`?: integer | revision, frame, the changes after `since` and the attempts refused since, each with its origin |
| `compose` | `text`?: string, `edits`?: [{statement \| member, text}], `expect`?: string | events, version |
| `sync` |  | events |
| `prepare` | `source`: string | id, manifest; or an error at a line and column |
| `publish` | `source`: string, `expect`?: "absent" \| "any" \| integer | id, generation |
| `retire` | `kernel`?: string | retired |
| `param` | `name`: string, `value`: value | name, value, events; a node's own value is refused, and changes by `compose` |
| `evaluate` | `nodes`?: [string], `settle`?: bool, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number | executed, reused, committed, outputs |
| `advance` | `frames`: integer, `fps`?: number, `until`?: {target, below \| above \| equals}, `save`?: {node.port: path}, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number | frames, stopped, target, value, saved, millis, report |
| `bench` | `frames`?: integer, `fps`?: number, `repeat`?: integer, `nodes`?: bool, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number | backend, frames, repeat, best_ms, median_ms, nodes, ran |
| `window` | `size`?: [w, h] \| null, `scale`?: number | the window |
| `read` | `output`?: string, `node`?: string, `port`?: string, `format`?: "rgba32f" \| "ppm" \| "field" | width, height, format, data (base64); words for a Field that is not pixels |
| `write` | `node`: string, `port`?: string, `data`: string, `format`?: "words" \| "field" | the state replaced |
| `snapshot` | `to`?: string | the world's revision and frame, every Kernel with its generation and source, every port's committed state; or the file written |
| `restore` | `snapshot`?: object, `from`?: string, `laws`: "current" \| "snapshot", `structure`?: "current" \| "snapshot", `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number | revision, frame, report |
| `reset` | `node`?: string, `port`?: string | what was reset |
| `help` | `section`?: string, `name`?: string | the index; a section's items; or one item in full. An unknown name is an error with `similar` |
| `trace` | `node`: string, `at`?: [x, y, z], `element`?: integer, `time`?: number, `rate`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `keys`?: [integer], `window`?: [w, h], `scale`?: number | kernel, at, bindings, results, calls |
| `pin` | `node`: string | pinned |
| `unpin` | `node`: string | unpinned |
| `quit` |  | bye |

## Transports

- **stdin/stdout.** `orla serve`, or the Player with `orla-player
  --serve <scene.osn>`: requests arrive while the window runs, `evaluate`
  answers with the next window frame under the person's pointer and
  keys, and `read` returns the frame the window shows.
- **Socket.** `--listen <host:port>` (`127.0.0.1:7788`; port 0 for any
  free port, named on stderr as `listening on ...`) accepts connections
  carrying the same lines, one response per request in order. Every
  connection is served by the one session; `quit` on a socket ends that
  connection, `quit` on stdin ends the process. See `bridges/blender`.
- **WebSocket.** A connection that opens with an HTTP upgrade carries the
  same lines as text frames (see `bridges/web`). A GET without an upgrade
  is answered 426, saying what it reached; any other HTTP request is
  answered 400 and closed. The upgrade is refused (403) except from a
  page served from this machine (`localhost`, `127.0.0.1`, `[::1]`), a
  page opened from a file (`null`), a client that is not a browser, or an
  origin named by `--allow-origin <origin>` (`https://host:port` as a
  browser writes it).
- **In process.** `orla_request` in
  [`orla.h`](../crates/interface/ffi/include/orla.h) (`include/orla.h` in a
  release) takes one line and
  returns its answer, beside the C functions that evaluate and read
  Fields in place.
- A socket request (a line or a WebSocket message) is at most 256 MiB; a
  longer one is answered with an error and its connection closed; the
  session serves the rest.

## Authority

- **Reads.** A served scene reads beside the documents of the scene as it
  first loaded (the scene, the graphs it places, what it uses), in every
  package, and in directories given with `--allow file.read=<dir>`. A
  composed document does not widen that; a file elsewhere is a
  `rejected` event naming it, and a `use` or `graph` naming one refuses
  the scene before any of it is read.
- **Writes** take `--allow file.write=<dir>`. Neither is ever granted
  over the wire ([decision](DECISIONS.md)).
- **Grants.** `status` answers `grants`, the capabilities a Kernel may be
  given: all, unless the server was started with `--grant` naming fewer.
  A Kernel requiring one not granted is refused when prepared, naming
  the ones it lacks; what ran before keeps running.

## Kernels

`prepare` and `publish` take a document holding one Kernel. The Kernel
sees the declarations the source makes or `use`s, then the scene's
shared ones, so a Kernel copied from a library file republishes as it
is. Kernels published through the protocol belong to no document; a
document that later defines the same id republishes it from the file.

`kernels` lists the backend's native operations alongside published
Kernels, at generation 0. A manifest's `memories` names ports that are
input and output at once, and `budget` the loop iterations one element
may run. A port's `type` is written as the Kernel writes it
(`Grid<Cell>`); binding is still decided by layout. `kernel` (id in the
`kernel` field; a request's `id` is the envelope's) returns a native
operation's contract where a Kernel shows its source; `publish` under a
native id and `retire` of one are refused. The source `kernel` returns is
what `publish` takes, for a node's inline Kernel too (the node's id
written in, its `domain` left out). `kernel` with a `generation` reads
the source a Kernel had then, while history remembers it.

## Scene

`document` answers the scene's `version`, a digest of its text, and with
a `version` the scene document a `scene` change loaded. `compose`
answers the version it wrote; with `expect` it writes only if the
document is still that version, else refuses with the current one.
Composing an old document against the standing version rolls the
structure back.

`compose` checks before it writes: a text that does not lower, names a
source nothing provides, wires a cycle, defines a Kernel that does not
compile or requires what is not granted, or uses or places a document
it may not read, is answered with a `scene_rejected` event as a sync
would be, and the document on disk and the running scene stay. A
document saved on disk whose own Kernels are refused does not replace
the running scene either.

The new text is `text`, whole, or `edits` of the document on disk, in
order, each on the result of the one before. An edit names one
statement by kind and name (`node`, `output`, `param`, `input`, `kernel`,
`struct`, `type`, `fn`, or `use` with its path) or, with `member`, one
member of a node's block, and gives its `text`: the statement or member
whole, declaring what the edit names, or `null` to remove it. What
exists is replaced; what does not is added after the last of its kind (a
member laid out as its block is). Only those bytes change; every other
statement, comment and blank line stays. A removed statement takes the
comment lines right above it. An edit that cannot be made (a text
declaring something else or more than one thing; nothing to remove)
refuses the whole `compose`, naming the edit, and writes nothing; the
result is then checked as any text is.

```json
{"op": "compose", "expect": "<version>", "edits": [
  {"node": "castle", "text": "node castle: castle.build@1 { base <- ground.result; }"},
  {"node": "view", "member": "voxels", "text": "voxels <- castle.result;"},
  {"node": "old", "text": null}
]}
```

`param` sets a parameter the scene declares (`param name: Type =
value;`), and only such: the session holds the value, the nodes naming
the param read it, and the document stays as written. A value a node
sets itself (`brush = 2.0;`) is the document's: `param` refuses its
name, naming the nodes that set it, and it changes by `compose` with the
node and `member`. A held value goes when the document stops declaring
the param.

## Values

A value in a request is JSON: a number (an integer is an `Int`, a
`Long` past 32 bits, and stands in for a float; a number with a fraction
is exact until its parameter gives the width), a string of digits for a
`Long` past 2^53, `true` or `false`, or an array of values in order,
taken in the parameter's shape: a vector's components, a matrix's
columns (`[[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]]` for
a `Mat4`), a struct's fields, an array's elements. A value of another
layout is refused with the parameter's type. Replies spell values the
same way (a `Double` as the number that reads back as it, a `Long` past
2^53 as its digits); `kernel` and `graph` name each parameter's type.

## Evaluation

`evaluate` takes `time`, `rate` (the sound device's clock; zero when
absent), `pointer`, `down`, `buttons` (held, by number: 0 left, 1
right, 2 middle), `wheel` (scrolled since the last evaluation), `keys`,
`window` (`[w, h]` in pixels) and `scale` (one when absent). An
evaluation naming no window sees the session's, set by the `window` op
(`size` as `[w, h]`, `scale`; `size` null for none) or `orla serve
--window <w>x<h>[@<scale>]`, and none otherwise. Inside the Player the
window is the person's and these two are ignored.

- `nodes`: evaluate only those nodes, what they read, the state (which
  must advance), and any output never produced; the rest keeps its last
  value.
- `settle: true`: first wait for every `async` node's run to land.

An evaluation's report carries:

- `timings`, `{node, millis}` for every node that executed, in order;
- `placements`, `{node, backend, routed, uploaded, downloaded}`: the
  backend that ran it, why not the main one (a scan on a GPU session, a
  `Double`), and the bytes brought to and from a device;
- `migrated`, `{node, port, kernel}` for every `<~` port carried into a
  new layout by its `migrate` Kernel;
- `reset`, `{node, port, reason}` for every such port restarted from its
  initial source;
- `pending`, the `async` nodes whose run is going on;
- `landed`, those whose run landed before this evaluation's nodes ran.

A failed evaluation names its `node` when the failure is one node's; a
failed migration names its `port` and `migration` too.

`advance` runs `frames` (1 to 100000) at `fps` (60 by default) from
`time`, with the rest of `evaluate`'s context held and `async` nodes
settled after each frame.

- `until`, `{target, below | above | equals}`, stops when the value says
  so; `target` is `node.port[index].field` (a struct field by name, a
  vector component by letter), as a replay writes it; the value is a
  Float, Int or Bool.
- `save`, an object of `node.port` to a file path (relative to the
  scene), writes those outputs when the run stops, by `run --save`'s
  rules, only into a `--allow file.write` directory; a path outside one
  is refused before any frame runs.
- Only what `until` and `save` name is materialized; with neither, the
  whole scene runs.
- The answer: `frames` (how many ran), `stopped` (`"frames"` or
  `"until"`), `target`, `value` (last seen), `saved`, `millis`, and
  `report` (the last frame's). A failure says `frame`.

`bench` runs `frames` (30 by default) at `fps` (60), `repeat` times (3),
and reports the least, as `orla bench` does; `frames` times `repeat` is
at most 100000. The first frame is discarded. With `nodes: true`, one
more run times every node, the device waited for after each; a node
placed several times is one line. The answer: `backend`, `frames`,
`repeat`, `best_ms`, `median_ms`, `nodes` (each `{node, ms, backend,
routed, uploaded, downloaded}`, costliest first), and `ran`, false when
the scene does not change with the clock and every node was reused after
the first frame. The clock advances as in `advance`.

`trace` runs one element of a node on the interpreter, compiled without
the optimizer so calls and `let`s are the source's. It takes `node`, and
`at` (`[x, y, z]`, a map element) or `element` (a scan step), with
`evaluate`'s context fields. The scene is evaluated first under that
context, the node's sources pinned; the trace is of the evaluation that
would follow (its `<~` inputs the state just committed), and commits
nothing. The answer: `kernel`, `at`, `bindings` and `results` (each a
list of `{name, type, value, ran}`, a struct as an object by field,
`ran` false for a `let` in a branch not taken), and `calls` (each
`{function, bindings, results}` in return order; `calls_truncated` past
256).

A `graph` answer shows a `<~` binding's migration as `migrate`, a node's
`async`, and each node's `backend` and `routed` (why not the main
backend, or null).

## State

`read` returns an output (`rgba32f` or `ppm`, base64); with `format:
"field"`, any Field in the Host's file format, which `write` with the
same format takes back, checked against the port's layout and extent.
`read` with a `node` returns a non-pixel Field as its words; `write`
with a `node`, a `port` and `data` (base64 words in the same layout)
replaces that port's committed state for the next evaluation's `<~`
readers. The port must be state, read by some `<~` binding, and the
word count must match. Words are the Kernel's, one per scalar (two for a
`Double` or `Long`, low first); a narrow Field is read widened and
written narrowed by the Host's rule, and only the `field` format carries
its packed bytes. After a failed evaluation, `read` shows the output it
computed.

`reset` forgets committed state; the next evaluation's `<~` readers
restart from their initial sources. With a `node` (and a `port`,
`result` by default) it forgets that port alone, which must be state. A
run off the frame that began from forgotten state, or from what was
computed from it, is let go; other runs go on. History records `{kind:
"reset"}`, with `node` and `port` for one.

`snapshot` returns the world for its caller to keep: revision and frame,
every Kernel with its generation and source, and every port's committed
state (`layout`, `shape`, `extent`, `data` as base64 words), under
`"snapshot": 1`, the format version. Only committed state is taken. With
`to`, it is written to that file, where `--allow file.write` allows.

`restore` takes a snapshot as `snapshot` gave it, or `from` a file the
session may read. `laws` is required: `"current"` puts the snapshot's
state under the running Kernels, migrated where a layout changed,
restarted where nothing carries it; `"snapshot"` first publishes the
snapshot's Kernels back, together, where their sources differ.
`structure` (`"current"` by default) with `"snapshot"` first writes back
the snapshot's scene document, checked as `compose` checks it, with its
Kernels in place. A state port the snapshot does not name restarts from
its initial source. The world is then proved by one frame, under the
context given as `evaluate` takes it, which commits nothing: on success
the world is the snapshot's and the next `evaluate` follows it; on
failure state, laws (generations too), structure and revision are put
back, and the attempt is in `history`. Runs off the frame are let go
either way ([decision](DECISIONS.md)).

## Status and history

`status` answers the backend, scene, watched files, counts, `versions`
(the release, ABI, language, IR and package format, as `orla version`
prints them),
`grants`, and the position: `revision` moves when the world changes
other than by running (a Kernel published or retired, the scene read
again, a param set, state written or reset); `frame` counts evaluations
committed since.

`history` (`since` a revision) says what made each later revision and
its origin: a document on disk, or the protocol and its channel
(`stdin`, `socket 3`, `websocket 4`, `orla_request`). The origin is how
a change arrived, not who sent it: an edit found on disk while a request
is answered is its document's; the scene a `compose` or `param` loads is
the request's. Refused attempts are listed with why and move nothing.
History is bounded and holds changes, never whole worlds.

## Help

Orla describes itself, from the tables its parts enforce: the builtins
and the forms from the compiler, the replay forms from the replayer, the
operations from the table above. An agent asks in three steps, each
answer small:

- `{"op": "help"}`: the index. Its `sections` (`builtins`, `kernel`,
  `scene`, `replay`, `protocol`), how many items each holds, and the
  `versions` the answer holds for.
- `{"op": "help", "section": "replay"}`: each item's `name` and one `line`.
- `{"op": "help", "name": "window"}`: every item of that name in full, in
  `found`, each with its `section`: a builtin's signatures, `requires` and
  whether only an entry may call it; a form's shape; a replay step's
  shape and examples; an operation's fields with their types and whether
  each is needed.

An unknown section or name is an error naming the near one in `similar`.

## Timing of changes

A saved edit to a running Kernel is prepared on a worker thread: `sync`
reports `preparing` at once and `published` (or `rejected`) on a later
`sync`; `evaluate` meanwhile runs the previous version. A scene whose own
Kernels are being prepared waits for them: its `scene_loaded` comes on
the `sync` that publishes them, `evaluate` until then runs the previous
scene on its Kernels, and a refused Kernel refuses the scene
(`scene_rejected`) with none of its Kernels published. `sync` reports
`routed`, `{node, backend, why}`, once when a node's placement off the
main backend begins or changes. A Kernel nothing runs yet prepares
inline.

`compose` and `publish` answer with their outcome: the scene landed or
refused, the Kernel published or refused. Where preparation is off the
frame (`serve`, the Player) the answer waits for it and nothing else
does: frames go on, other connections are answered, and what a
connection sent after it is answered after it, in order. One change is
made at a time. `orla_request` answers in its call.
