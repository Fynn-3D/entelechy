# Creation protocol

**Answers:** what an agent sends over the wire, and what comes back.

`entelechy serve <scene.esn>` drives a live session over standard input and
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

## Operations

A request is one JSON object a line, with an `op` and an optional `id`
echoed back. An answer is one line, `{"id", "ok": ...}` or `{"id",
"error": {"message", "line"?, "column"?}}`. An error carries what acts
on it as data: an unknown operation the `operations`, an unknown `help`
name the `similar` ones, a failed evaluation its `node` (a state
migration's also `port` and `migration`) and a run's `frame`. A field
marked `?` may be left out. Every field is held to its type before its
operation reads it: a request with a field of the wrong type, one its
operation does not take, or without one it needs, is refused with the
fields it takes, the `field` at fault and the type `expected`.

What a request and an answer carry, and the doors a session is reached
through:

<!-- rendered: wire -->
| part | in | type | says |
| --- | --- | --- | --- |
| `op` | request | string | The operation: one of the protocol section's, each with its own fields. |
| `id` | request | value | Anything; the answer carries it back as it was sent, to pair them. |
| `id` | answer | value | The request's `id`, or null when it sent none or was not JSON. |
| `ok` | answer | object | What the operation answered, as its entry in the protocol section says. |
| `error` | answer | object | Why the request was refused, and what acts on it as data; the world is as it was. |
| `message` | error | string | Why, one or more sentences for a person or an agent to read. |
| `line` | error | integer | Where the fault is, from 1: in the text the request carried, or in `path`. |
| `column` | error | integer | Where on that line, from 1. |
| `similar` | error | [string] | The names nearest an unknown one. |
| `fix` | error | object | The change that mends the fault, when one does: an item of `compose`'s `documents` that takes its document's place, handed back as it is; `path` with `text`, the refused change's own document with a `use` item added, or with `edits` adding it to the document on disk. A `rejected` event carries it too. |
| `fields` | error | [string] | The fields the operation takes. |
| `field` | error | string | The field at fault. |
| `expected` | error | string | The type that field is. |
| `operations` | error | [string] | Every operation, for an unknown one. |
| `version` | error | string | The scene document's version as it stands, for a compose refused. |
| `path` | error | string | The scene document a refused compose would have written, or the document a failed element's operation is written in. |
| `written` | error | bool | Whether a compose wrote its document: false when refused. |
| `text` | error | string | The document an `edits` compose made, which `line` counts in. |
| `node` | error | string | The node an evaluation failed at. |
| `port` | error | string | The state port whose migration failed. |
| `migration` | error | string | The Kernel that failed to migrate that state. |
| `frame` | error | integer | The frame a run failed at, counted from its first. |

| door | opened by | carries |
| --- | --- | --- |
| stdin | `entelechy serve <scene.esn>` | Request lines on standard input, answer lines on standard output; `quit` ends the process. |
| socket | `entelechy serve <scene.esn> --listen <host:port>` | The same lines over each TCP connection to the address; `quit` ends the connection. |
| websocket | `entelechy serve <scene.esn> --listen <host:port>` | The same lines as WebSocket text messages to the same address, one request a message. |
<!-- end -->

| op | fields | answer |
| --- | --- | --- |
| `status` |  | backend, scene, watched, kernels (published Kernels), natives (the backend's native operations), nodes, loaded (whether a scene is), refused (why not, when not), revision (moved by every change but running), frame (the frames committed since that revision), fusion, grants, memory (`used` and `budget`, the Host's bytes; `cold`, the nodes whose outputs are let go, made again when read; where the bytes are: `host`, `device`, `kept` (resident Fields' Host copies), `spare`, `loose` (Fields nothing holds), `backends` (what each keeps of its own) and `top`, the largest Fields with what holds each), versions (`entelechy`, `api`, `language`, `ir`, `package`) |
| `kernels` |  | a list of manifests, each with its generation, which `publish`'s `expect` names |
| `kernel` | `kernel`?: string, `generation`?: integer | manifest, generation, source, as `publish` takes it |
| `graph` |  | inputs, params, nodes, outputs (each with its `source`, and `to`: the Host device it is given to, or null), fonts (each font node: id, path, codes, its outputs and the inputs they are in), files (each files node: id, path, paths, its outputs and the inputs they are in), fields (each fields node: id, type, path, paths, its outputs and the inputs they are in) |
| `document` | `version`?: string, `path`?: string | path, text, and version, which `compose`'s `expect` names |
| `history` | `since`?: integer | revision, frame (the frames committed since that revision), the changes after `since` and the attempts refused since, each with its origin |
| `compose` | `text`?: string, `edits`?: [{node?: string, output?: string, param?: string, require?: string, input?: string, kernel?: string, struct?: string, type?: string, fn?: string, use?: string, member?: string, text: string \| null}], `expect`?: string, `path`?: string, `documents`?: [{path?: string, text?: string, edits?: [{node?: string, output?: string, param?: string, require?: string, input?: string, kernel?: string, struct?: string, type?: string, fn?: string, use?: string, member?: string, text: string \| null}], expect?: string}] | events, version, path (the scene), written; with `path` or `documents`, documents (each written document's path and version) |
| `sync` |  | events |
| `prepare` | `source`: string | id, manifest; or an error at a line and column |
| `publish` | `source`: string, `expect`?: "absent" \| "any" \| integer | id, generation; and when it computes a `<~` port's initial source, `keeps` (those `node.port`s, which keep what they committed) and `reset` (the requests that start them from the new law) |
| `retire` | `kernel`?: string | retired |
| `param` | `name`: string, `value`: value | name, value, events; a node's own value is refused, and changes by `compose` |
| `evaluate` | `nodes`?: [string], `steps`?: integer, `dt`?: number, `until`?: string, `settle`?: bool, `measure`?: bool, `clock`?: bool, `time`?: number, `rate`?: number, `queued`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `pads`?: integer, `keys`?: [integer], `text`?: string, `window`?: [w, h], `scale`?: number | executed, reused, committed, outputs, timings, measured, device, let_go (nodes let go to make room in the Host); when `steps` or `until` is given, steps, active, time, millis, stopped (`"steps"` \| `"until"` \| `"settled"` \| `"nonfinite"`), until; in the Player, ignored (the context fields it gave, which the window's own input stands in for) and a note |
| `read` | `output`?: string, `node`?: string, `port`?: string, `format`?: "png" \| "rgba32f" \| "ppm" \| "summary" \| "words" \| "field", `from`?: integer, `count`?: integer | a picture: width, height, format, data (base64). Any other Field: shape, extent, count, schema, `fields` (each number's name in an element of several, `hp`, `pos.x`), and its summary (each number's min, max, min_at, max_at, sum, mean, rms, how many were not zero, and when any were not finite: nonfinite, nan, inf, first_nonfinite), or `values` from `from` (a number, or a list for an element of several), or format and data (base64) |
| `write` | `node`: string, `port`?: string, `data`: string, `format`?: "words" \| "field" | the state replaced |
| `snapshot` | `to`?: string | the world's revision and frame (the frames committed since it), every Kernel with its generation and source, every port's committed state; or the file written |
| `restore` | `snapshot`?: object, `from`?: string, `laws`: "current" \| "snapshot", `structure`?: "current" \| "snapshot", `time`?: number, `rate`?: number, `queued`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `pads`?: integer, `keys`?: [integer], `text`?: string, `window`?: [w, h], `scale`?: number | revision, frame, report |
| `pause` | `paused`?: bool, `step`?: integer | paused, time (the scene's, in seconds); refused where the caller gives the time (`serve`: `evaluate`'s `time`) |
| `reset` | `node`?: string, `port`?: string | what was reset |
| `help` | `name`?: string | `reference`, the whole of it as one Markdown text, and `versions`; or with `name`, every item of that name in full. An unknown name is an error with `similar` |
| `trace` | `node`: string, `at`?: [x] \| [x, y] \| [x, y, z], `element`?: integer, `time`?: number, `rate`?: number, `queued`?: number, `pointer`?: [x, y], `down`?: bool, `buttons`?: [integer], `wheel`?: [x, y], `sticks`?: [[x, y]], `pads`?: integer, `keys`?: [integer], `text`?: string, `window`?: [w, h], `scale`?: number | kernel, at, bindings, results, calls, and when the element failed, fault: its message, the function it failed in with that function's bindings, and the operation's path, line and column |
| `pin` | `node`: string | pinned |
| `unpin` | `node`: string | unpinned |
| `quit` |  | bye |

## Transports

- **stdin/stdout.** `entelechy serve`, or the Player with `entelechy-player
  --serve <scene.esn>`: requests arrive while the window runs, `evaluate`
  answers with the next window frame under the person's pointer and
  keys (its own input fields are not read, and are named in its answer's
  `ignored`; `measure` is, and times that frame), and `read` returns the frame the window shows.
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
  client that is not a browser, or an origin named by `--allow-origin
  <origin>` (`https://host:port` as a browser writes it). A page opened
  from a file says `null`, as a sandboxed frame on any site does, so it
  connects only with `--allow-origin null`.
- **In process.** `entelechy_request` in
  [`entelechy.h`](../crates/interface/ffi/include/entelechy.h) (`include/entelechy.h` in a
  release) takes one line and
  returns its answer, beside the C functions that evaluate and read
  Fields in place.
- A socket request (a line or a WebSocket message) is at most 256 MiB; a
  longer one is answered with an error and its connection closed; the
  session serves the rest. A connection is read no further while 64 of
  its answers wait to be written, so a peer that does not read its
  answers holds back only itself.

## Authority

- **Reads.** A served scene reads beside the documents of the scene as it
  first loaded (the scene, the graphs it places, what it uses, the files
  its inputs read), in the directories packages are found in by name, and
  in directories given
  with `--allow file.read=<dir>`, judged where a path lands after its
  links and `..`. Every read the session makes is held to that, however
  deep the document that names the file. A composed document does not
  widen it; a file elsewhere is a `rejected` event naming it, and a `use`
  or `graph` naming one refuses the scene before any of it is read.
- **Writes** take `--allow file.write=<dir>`. Neither is ever granted
  over the wire ([decision](DECISIONS.md#what-a-served-scene-reaches-is-what-the-person-running-it-gave)). `compose` writes only the
  scene's own documents: `.ent`, `.esn` and `.replay` files under the
  scene's directory, where they land after their links, none in a package
  directory; the person who serves a scene gave its directory.
- **Grants.** `status` answers `grants`, the capabilities a Kernel may be
  given: all, unless the server was started with `--grant` naming fewer.
  A Kernel requiring one not granted is refused when prepared, naming
  the ones it lacks; what ran before keeps running.

## Kernels

`prepare` and `publish` take a document holding one Kernel. The Kernel
sees the declarations the source makes or `use`s, then the scene's
shared ones, so a Kernel copied from a library file republishes as it
is. `publish` changes the running world, not a file: a Kernel published
through the protocol belongs to no document, the document it came from
is left as it is on disk, and when that document next changes, its
Kernel is published from the file again. A Kernel is kept by writing it
into a document the scene uses (`compose` with that document's `path`),
or with the whole world by `snapshot`.

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
a `version` a text the session remembers: a document of the scene as a
`scene` change loaded it, or as a `compose` wrote it, edited on disk or
over the protocol alike. `compose`
answers the version it wrote; with `expect` it writes only if the
document is still that version, else refuses with the current one.
Composing an old text against the standing version rolls that document
back.

The scene's other documents are composed alike. `path` names one, from
the scene's directory: a document the scene uses or places, or a new
one; `document` with that `path` reads it. `documents` changes several
at once, each with its own `path` (none for the scene), `text` or
`edits`, and `expect`: a new Kernel document and the scene's `use` of
it, the node that runs it and the output it feeds, in one change. All
are checked together, as `entelechy check` checks the scene with them,
their Kernels compiled and granted, before any is written; one refused
writes none, and the refusal names the document at fault in `path`,
with its `version` and, for `edits`, the `text` they made. A document
written must be one the scene, as written, uses or places. The answer
lists each document written with its new version, under `documents`,
once the Kernels it changed run.

A replay is one of the scene's documents too, its tests: `compose` writes
a `.replay` under the scene's directory whole, by `text`, once it parses
and the scene (and the Kernels) it names are there, written or written in
the same compose; no scene uses a replay, so none must. What it wrote
plays as any replay does (`entelechy replay`, the Player's `--replay`).

`compose` checks before it writes: a text that does not lower, names a
source nothing provides, wires a cycle, defines a Kernel that does not
compile or requires what is not granted, or uses or places a document
it may not read, is refused as an error, as a refused `publish` is: why,
at its `line` and `column` where it has a place, the document's
`version`, and for `edits` the `text` they made, which the line counts
in. The document on disk and the running scene stay. A
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
comment lines right above it; a text ending in a comment takes the place
of the comment that ended the line of what it replaces, and one beginning
with comment lines the place of those right above it. An edit that cannot be made (a text
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
`text` (the characters typed since the last evaluation, what `typed()`
reads; more than 32 are refused, the rest is the next evaluation's to
give), `window` (`[w, h]` in pixels) and `scale` (one when absent). An
evaluation naming no window sees the one `entelechy serve --window
<w>x<h>[@<scale>]` gave, and none otherwise. Inside the Player the
window is the person's and these two are ignored. Many frames are many
evaluations, each at its own `time`.

- `nodes`: evaluate only those nodes, what they read, the state (which
  must advance), and any output never produced; the rest keeps its last
  value.
- `settle: true`: first wait for every `async` node's run to land.

An evaluation's report carries:

- `timings`, `{node, millis}` for every node that executed, in order:
  on a device, the time to submit its work, as the device runs it after;
  with `measure: true` (then `measured: true`), each node waits for the
  device, and its time is the work's, by the device's own clock where it
  keeps one; measuring stalls the frame at every node, so it is for
  finding where time goes;
- `placements`, `{node, backend, routed, uploaded, downloaded}`: the
  backend that ran it, why not the main one (a scan on a GPU session, a
  `Double`), and the bytes brought to and from a device;
- `migrated`, `{node, port, kernel}` for every `<~` port carried into a
  new layout by its `migrate` Kernel;
- `reset`, `{node, port, reason}` for every such port restarted from its
  initial source;
- `pending`, the `async` nodes whose run is going on;
- `landed`, those whose run landed before this evaluation's nodes ran;
- `refused`, `{node, members, reason}` for every group the backend took
  Kernel by Kernel but refused fused, said once: its members run apart
  until one of their Kernels changes;
- `let_go`, how many nodes had their cached outputs let go to make room
  in the Host: a later frame passes one by while its key is unchanged and
  makes it again for a reader that runs, so above zero frame after frame
  the scene holds more than `--memory` allows (`status` answers `memory`,
  the bytes held of the budget, and the nodes let go).

A failed evaluation names its `node` when the failure is one node's; a
failed migration names its `port` and `migration` too. An element that
failed (integer arithmetic, a loop budget) is traced again: when the
trace finds the failing operation (for a budget, the loop that ran the
most iterations) and a document writes the Kernel, its `path`, `line`
and `column` come with the error.

`trace` runs one element of a node on the interpreter, compiled without
the optimizer so calls and `let`s are the source's. It takes `node`, and
`at` (`[x, y, z]`, a map element) or `element` (a scan step), with
`evaluate`'s context fields; with none of them, the context the last
frame was evaluated under (a window's size, scene time and input as that
frame saw them). What the node reads is evaluated first
under that context, its sources pinned and the node itself not run; the
trace is of the evaluation that would follow (its `<~` inputs the state
last committed), and commits nothing. The answer: `kernel`, `at`, `bindings` and `results` (each a
list of `{name, type, value, ran}`, a struct as an object by field,
`ran` false for a `let` in a branch not taken), and `calls` (each
`{function, bindings, results}` in return order; `calls_truncated` past
256). An element that fails is traced up to its failure: what ran before
it is there, the rest `ran` false, and `fault` says why (`message`), the
`function` it failed in with that function's `bindings` as it failed,
and where the operation is written (`path`, `line`, `column`).

A `graph` answer shows a `<~` binding's migration as `migrate`, a node's
`async`, and each node's `backend` and `routed` (why not the main
backend, or null).

## State

`read` answers what it is asked, and without asking, how the Field
looks: a picture as a PNG, any other
Field as its summary, each number's least, most, sum and mean over
every element, and how many were not zero. `from` and `count` answer up to 4096 elements as numbers, an
element of several as a list in order. The whole Field comes when a
`format` names it: a picture's `rgba32f` or `ppm`, or any Field's
`words`, or `field`, the Host's file format, which `write` with the same
format takes back, checked against the port's layout and extent. `write`
with a `node`, a `port` and `data` (base64 words in the same layout)
replaces that port's committed state for the next evaluation's `<~`
readers. The port must be state, read by some `<~` binding, and the
word count must match. Words are the Kernel's, one per scalar (two for a
`Double` or `Long`, low first); a narrow Field is read widened and
written narrowed by the Host's rule, and only the `field` format carries
its packed bytes. After a failed evaluation, `read` shows the output it
computed, and a read of one it kept from being made is told the failure:
the node that failed, why, and the nodes run fused into it.

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
either way ([decision](DECISIONS.md#reality-changes-atomically-history-records-why)).

## Status and history

`status` answers the backend, scene, watched files, counts, `versions`
(the release, ABI, language, IR and package format, as `entelechy version`
prints them),
`grants`, and the position: `revision` moves when the world changes
other than by running (a Kernel published or retired, the scene read
again, a param set, state written or reset); `frame` counts evaluations
committed since.

`history` (`since` a revision) says what made each later revision and
its origin: a document on disk; the scene itself, `{"output": name}`,
for a document its output `name` wrote and that is as it wrote it (a
law the world wrote); or the protocol and its channel
(`stdin`, `socket 3`, `websocket 4`, `entelechy_request`). The origin is how
a change arrived, not who sent it: an edit found on disk while a request
is answered is its document's; the scene a `compose` or `param` loads is
the request's. A `compose` is one `documents` change listing each
document it wrote with its version; a `scene` change lists, under
`documents`, the scene's other documents it loaded at a version not
remembered before. `document` reads each back. Refused attempts are
listed with why and move nothing.
History is bounded and holds changes, never whole worlds.

## Help

`help` answers Entelechy's syntax and API, layer by layer, gathered from the
tables the implementation executes: the types, the builtins and the
forms from the compiler, the reserved words from the parser, what each
capability grants from the ABI, a package's parts from the session, the
operations and the wire above, the replay lines from the replayer.

- `{"op": "help"}`: `reference`, all of it as one Markdown text
  ([REFERENCE](REFERENCE.md) is the same text), and the `versions` it
  holds for.
- `{"op": "help", "name": "window"}`: every item of that name in full, in
  `found`, each with its `section`: a builtin's signatures, `requires` and
  whether only an entry may call it; what a capability grants and the
  builtins that need it; a form's shape; a replay line's shape and
  examples; an operation's fields with their types and whether each is
  needed.

An unknown name is an error naming the near one in `similar`. Whole
programs are in `examples/`.

## Timing of changes

A saved edit to a running scene's documents is read and compiled on
another thread, and a running Kernel's replacement prepared on a worker
thread: `sync` reports `preparing` once its document is compiled (a
fault there as `rejected`), and `published` (or `rejected`) on a later
`sync`; `evaluate` meanwhile runs the previous version. A scene whose own
Kernels are being prepared waits for them: its `scene_loaded` comes on
the `sync` that publishes them, `evaluate` until then runs the previous
scene on its Kernels, and a refused Kernel refuses the scene
(`scene_rejected`) with none of its Kernels published. `sync` reports
`routed`, `{node, backend, why}`, once when a node's placement off the
main backend begins or changes; `large`, `{path, id, why}`, with each
version published of a Kernel that carries more through its branches
than a device's compiler takes in comfort (what `entelechy check` says of
it); `idle`, `{path, ids}`, the Kernels a document of the scene
publishes and no node uses, once and again when they change; and
`unrun`, `{node, why}`, a node with no `domain` whose Kernel compares
`element()` with its first input's count, so the elements past that
count are never run (what `entelechy check` says of it), once and again when
it changes. A Kernel
nothing runs yet prepares inline.

`compose` and `publish` answer with their outcome: the scene landed or
refused, the Kernel published or refused. Where preparation is off the
frame (`serve`, the Player) the answer waits for it and nothing else
does: frames go on, other connections are answered, and what a
connection sent after it is answered after it, in order. One change is
made at a time. `entelechy_request` answers in its call.
