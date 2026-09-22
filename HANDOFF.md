# Orla Handoff

## Context

Conversation is Chinese; source, docs, and commits are English. `main` is
the line, tagged `v0.1.0` at the checkpoint after the structure round; a
round of work is a short branch off it (`feat/…`, `refactor/…`, `docs/…`),
verified, then fast-forwarded back, one commit per block, a merge commit
only when a fast-forward cannot do. Merging back is said once per round;
do not push. Commit as the
configured identity. Preserve both architectural bibles verbatim (`Orla Architecture
Plan.md`, `Orla — Minimal Core Architecture.md`); their handle lists stay as
written while the code has one `Handle`.

The previous implementation at `../bak/orla` is a read-only reference.

Layering as decided: the runtime crates and the `orla` CLI take no external
dependencies; the interface layer (L8: `cli`, `player`, `import`, `export`,
`jit`, `ffi`) is the only place that may (player: winit, softbuffer, cpal,
gilrs; gpu: wgpu; import: symphonia; export: none; jit: cranelift 0.131,
the newest that builds on Rust 1.93; ffi: none). `jit` and `ffi` are the
two crates with `unsafe`. Crates are named by what they are, not by
layer. Bridges live in `bridges/<program>/` in that program's language, outside
the workspace, speaking only the protocol and the Host's file formats.

Decisions not recoverable from code: lossy audio is decoded (importer) but
never encoded (WAV is the way out; ffmpeg compresses); JPEG is not decoded
(convert to PNG); interlaced PNG is refused; `.wav` saves at 48 kHz; a node's
`times` is structural, so a param it names lowers the document again; `sample`
stays in `compute` while functions with Field parameters use `at`/`at_linear`;
text is a convention of the shared library, not a contract of the runtime:
`packages/text.orlapkg` declares the glyph row and the drawing, and the
font atlas beside it is baked from rows of bits in `font.orla` (a face is
replaced by re-baking or by another PNG in the same 6x8 cells); a scatter's
order is its determinism rule (sources applied one after another in element
order, floats included), which the GPU honours by applying the pairs in one
invocation after a parallel map, rather than with atomics: exact and slow
for very many sources, and a place to optimise later without changing any
result (int `sum`, `min`, `max`, and `last` by index are order-free); the
the CPU compiles a module to word code on every execution because that
was measured at 2 to 40 microseconds per Kernel against milliseconds of
running (voxel, raymarch, life), so no compiled program is cached across
frames; a headless session's window is the session's (`serve --window`,
the `window` op) unless an `evaluate` names one, and the Player's is the
person's;
scatter's `min` and `max` order floats by their bits' total order on both
backends so the GPU may use atomics, and a float `sum` is a bitonic sort
of (target, source) followed by a per-target fold in source order, all
sort steps in one compute pass (dispatches in a pass see each other's
writes in order; a pass per step cost ten times more), which is exact
and, at a hundred thousand sources into 320x240, under 2 ms on the GPU
against 38 on the CPU, where the walk in one invocation took about 15;
the
window is context under `input.read` rather than a capability of its own
(its size is in physical pixels, its scale factor is what a bar laid out
in logical pixels divides by), the Player reports a default 960x540 window
before its own opens, and a resize needs no structural step because the
plan is drawn every evaluation and the extent is in the cache key; the
listener tells a WebSocket from a line client by the first four bytes
(`GET `) and never serves a page, so a bridge's files are opened from
disk, and `orla.js` is a classic script because browsers refuse module
imports between file URLs; a scan's step loop on the GPU has a
`storageBarrier()` between steps because the Metal compiler hoisted a
step's first memory load out of the loop (every step read the starting
value; the CHIP-8 escaped it by writing before reading); the JIT is on Cranelift 0.131 (the newest that
builds on Rust 1.93), returns every function's result through a
caller-provided stack slot (arm64 refuses a struct's worth of return
registers), keeps `unsafe` to the one crate that calls the code it made
(its own `[lints]` instead of the workspace's `forbid`), and prepares one
compiled module per Kernel version, freed with it; a reduction's `merge`
is compiled with its two values passed through pointers (its words vary
by Kernel, a native call from Rust cannot), and the fold and a scatter's
apply are `orla-rules`' loops so the three backends stay one rule (that
crate holds every rule the backends share: invocation checks, scan and
scatter sizing, addresses, combines, float key order, the matrix
formulas, the rasterizer's contract; the CPU is the reference
implementation, not the home of the law); the
C ABI (`orla-ffi`) is blocking and single-threaded per session, its
strings owned by the session until the next call and a Field's words
until the next evaluate, request or close (the Host may release them),
its header written by hand rather than generated so no build dependency
comes in, and the whole protocol reachable through one `orla_request`
so the fast path stays ten functions; a trace is of the evaluation
that would follow the one it runs first (the node's `<~` inputs are the
state just committed) because the inputs of the evaluation just done are
gone once it commits, and the trace commits nothing itself.

Before a commit run, in this order: `cargo fmt --all`, `cargo clippy
--workspace --all-targets -- -D warnings`, `cargo test --workspace` (about 40 s;
the example-replay test is the slow one; the feature matrix in
`crates/matrix` runs every feature under every mode on the JIT and the
GPU against the CPU, so a new language feature goes into
`orla_matrix::features` and a new mode into `modes`), and every replay on the three backends:
`for f in examples/*/tests/*.replay; do for b in gpu cpu jit; do
./target/release/orla replay $f --backend $b; done; done` (about 20 s after
`cargo build --release`). Tests in one binary run in parallel: a test that
writes files takes a directory of its own (`scratch_in`, or a named scratch).

The Blender bridge is tested headless: `orla serve examples/mesh/scene.osn
--listen 127.0.0.1:0` (port on stderr) then
`/Applications/Blender.app/Contents/MacOS/Blender -b --python
bridges/blender/test_bridge.py -- --port <port> --out <dir>`. The web
bridge is tested by `node bridges/web/test.mjs` after `cargo build
--release`: it serves the text and voxel examples itself and drives
`bridges/web/test.html` in a headless Chrome over the DevTools protocol
(the `--dump-dom` shortcut does not wait for asynchronous work in the
new headless mode, which is why a runner exists). The C ABI is tested by
`cargo test -p orla-ffi` (in the workspace run) and by hand with
`cc bridges/c/hello.c -Icrates/ffi/include -Ltarget/release -lorla_ffi`
then `DYLD_LIBRARY_PATH=target/release ./a.out examples/text/scene.osn
jit`, which prints the picture's size, a pixel before and after a
publish. The Godot bridge is
tested by `godot --headless --path bridges/godot -s test.gd` after
`cargo build --release` (`/usr/local/bin/godot`, 4.6): the script spawns
the text, voxel and mesh sessions through `OS.execute_with_pipe`, reads
each port from stderr, and ends with `RESULT: ok`; its scripts reach
each other by `preload`, since a headless `-s` run has no global class
cache. The user's own
Blender runs an MCP add-on on 127.0.0.1:9876 that executes Python sent as
`{"type":"execute","code":...,"strict_json":false}` terminated by `\0`,
answering the same way; the bridge was driven live through it (push with Live,
pull, Watch, publish, `param level`). A play session's Player and scratch
files may be left running or gone; nothing of it is in the repository.

## Next

Against the architecture plan's phases (sections 37 to 42): phases 1
and 2 are done and exceeded; phase 3's surfaces (the creation protocol
with positioned diagnostics, preview, atomic publish, compose, `help`,
the registry) are done and its demonstration is on record as the
`scanline` example, an agent's `agent.jsonl` replayed by a protocol
test; phase 4 has fusion, Metal through wgpu, a JIT the plan never asked
for, and state migration (section 24: `<~ … migrate id@v`, a map
Kernel run once over the old state when a republished Kernel changed
its layout; without one the port restarts from its initial and says
so; a failing migration leaves the state as it was), and an IR optimizer (exact rewrites only: construction/extraction
cancelling, literal folding without failures, common subexpressions
scoped by block, dead values; 25 to 30 percent fewer instructions, the
interpreter a fifth to two fifths faster, the JIT where Field reads
repeated), and packages (section 43: a directory with a `package.osn` manifest
naming the package and the least language and IR it needs, used by a
scene like any library, carried as one `.orlapkg` file by `orla
pack`/`unpack`; with section 44's four versions, `orla version`), and
async Kernels (`async = true;` on a node: its run goes on a thread of
its own over copies of its inputs, the frame carries its last output and
holds its state until the run lands, `settle` waits; the `async`
example steps the Thumb machine half a million instructions a step
while the picture keeps its pace). The plan's parallel scheduler was
measured and declined: a frame is the sum of its nodes on all three
backends (fluid on the GPU: 5.8 ms of nodes in a 7.7 ms frame, the rest
per-node dispatch), a map already uses every core on the CPU and the
JIT, and the GPU queue is in order, so threads across nodes would gain
nothing; an `async` node is the one case that pays, independent work
that need not finish inside the frame. The Player's F10 line shows
the frame's true cost (the interval between frames, since the picture
is presented on the device without a readback), the evaluate and
present times, and what ran; the title is the scene's name.

Phase 4 is closed (2026-09-21) with WASM deferred until a browser use
case asks for it, by this assessment: as a backend (the plan's IR to
CPU/GPU/WASM diagram) it has no value here, since the JIT already
generates machine code through the same Cranelift a wasm engine would
use and the ABI is the sandbox (no pointers, clamped reads, granted
capabilities); as a target (Orla itself in a page) the core crates and
the interpreter would go in one round (a timer and a thread to guard by
target, wasm-bindgen, the protocol as a function call, the web bridge
then local), the JIT cannot (no executable memory in a page), and
WebGPU would take one or two more rounds because the gpu crate blocks
on device work at observation and a page cannot; without WebGPU a page
runs life and snake, not showcase (418 ms a frame on the interpreter
at 640x360 against 3.6 ms of GPU work).

Phase 5 (section 42, the general dynamic compute runtime) by domain,
what is there and what is open: image is the origin (blur, marching,
pixelfx; PNG, PPM and PGM in and out); audio has device-paced sound
outputs, `rate()`, WAV in and out (tone, slg); geometry has glTF in
and out, the raster contract, and the Blender and Godot bridges (mesh,
cube, model); simulation has fluid, particles, life, ant, and the
games; machines, which the plan does not list, have CHIP-8 and the
Thumb core; agents and automation have the creation protocol, its
three bridges and the C ABI, and a scriptable `run` and `replay`. Done
since: video's output half, `orla render` (N frames at a rate, the
clock the frame count over the rate, each picture a PNG in a directory
and the sound output one WAV beside them, `async` nodes settled every
frame; ffmpeg makes the video). Also done: AI's first example,
`learn`, a 2-16-1 network trained by gradient descent to tell the
inside of a circle from the outside, all of it Kernels (the gradient a
scatter, element `e` being weight `e % 65` of point `e / 65`, so no
array is summed; the step placed eight times a frame with `times`, the
weights carried on through `<~`; the loss a reduce; the CPU and the
JIT bit for bit, the GPU's sums within two percent after 160 steps),
which found that a reduce's output wants a declared `type`, that a
scatter maps over its first input's elements so a product-sized step
takes a `work` Field of that size, and that fixed loop counts were no
obstacle. Also done: video's input half, `input clip: Image = frames "dir/" at
N;`, a directory's files in name order as frames, the frame of the
moment brought in at each evaluation (the `clip` example), so a clip
goes ffmpeg, `frames`, `render`, ffmpeg. Then `rubik`, the AI demonstration the user asked for: a Rubik's cube
in three dimensions (54 stickers as a `Rows<Int>`, twelve turns as
permutations from a table generated by a cubelet model, the picture two
boxes cast per pixel with the turning layer in its own frame, the keys
and the pointer through `input.read`) and a 54-32-1 network trained
while it is played with, on 64 scrambles it makes itself every frame
(the hidden units and the deltas as maps, the gradient as a scatter
over cubes times weights with one chain-rule term per element, the
update a map, the weights carried through `<~`), its guess of the turns
from solved shown over the picture: training and inference live, the
CPU and the JIT bit for bit, the GPU within rounding, the replay at
1.5 s on the interpreter. Done since: `render --save`
keeps a node's output after the last frame, and `rubik/play.osn` reads
the weights it kept (300 frames of training) as an `input`, inference
as its own scene; `text.label` in the shared text library is a caption
and a number as one row of codes, so a readout is three nodes and no
Kernel. The assessment's odds and ends are done too: diagnostics name the file
that declares a name a Kernel cannot find and the `use` that would
bring it in, and two documents that use each other are refused by name;
`orla help` and the protocol's `help` carry every form of the language
beside the builtins; `run --profile` measures the device rather than
the submission; and the optimizer writes short functions out where they
are called, which took the interpreter's replay suite from 14.5 to
12.4 seconds and raymarch from 496 to 365 ms, while a trace compiles
the Kernel again without the optimizer so it still follows the calls.

The assessment's three next things are done:
the JIT is the default backend (the Player takes the GPU, then the JIT,
then the interpreter; `--backend cpu` is how the reference is asked
for), a loop's count may be any `Int` expression read once before the
loop and clamped to the bound, and a node may be a filled Field
(`node zeros: Rows<Float> { domain = 65; }`, `value = ...` for anything
but the element's zero), which took five zero-fill Kernels out of the
examples. A file may come in as its bytes, `input level: Rows<Int> = bytes
"level.bin";`, so a format of one's own is read by a Kernel rather than
by a tool: the `level` example walks a header, a tile map and an entity
list, and draws them, with no Rust and no importer. The division stands:
what other software writes has a reader in `orla-import`, where the
dependencies are; what is yours needs none.

A score goes in and out: `orla-import` reads a standard MIDI file into
rows of `{start, duration, pitch, velocity, channel}` in seconds, and
`orla-export` writes those rows back as a file a music editor opens,
both from the format's own definition and with no dependency; the
`score` example plays one and draws its piano roll. Writing the reader
found two faults in it, both caught by tests built from the format
rather than from the writer: a tempo change while a note sounds
underflowed the tick arithmetic, and a note's second must be taken when
it is struck.

A play round (the `path` tracer and the `spectrum`) turned up two
things and both are fixed: a function could not take a keyword's name
(`fn scatter(...)` was refused, and the message did not say why), and
now any word but the five that begin an expression names one, since a
call is a word and a `(` wherever it stands; and a replay's expectation
could name a struct's field but not a vector's component, which the
language allows, so `expect sum.result[0].a = 32` reads the alpha now,
and the protocol's `advance` watches the same spelling. The round also
caught a fault in its own example, a vibrato written as a factor on the
clock, which is a chirp: the replay found the pitch had moved.

Done since: the interpreter is faster by the two things that were left
to try. The optimizer lifts a loop's unchanging work out of it, which
every backend reads and which took raymarch on the interpreter from 365
to 246 ms and the replay suite from 12.4 to 11.3 s; and the interpreter
folds a pair of float operations into one instruction, worth another 3
to 9 percent (11.3 to 10.8 s). What is left of the gap to the JIT is
structural: every value goes through a frame slot, and the JIT keeps it
in a register. Wide interpretation, several elements through one
instruction stream, is the only thing that would close it, and it is a
second execution path, declined twice now.

Automation past one frame is the
protocol's `advance`, a count of frames in one request that stops when
a watched value says to and saves what it was asked for, demanding only
what it watches; an agent's training loop is one request per
checkpoint. Phase 5 has no open item of its own now: what is left is
the plan's own leftovers, WASM, and whatever a round of play turns
up. The assessment of the experience (2026-09-21) listed what was
awkward; taken in the same day: a scatter sets its own `domain` (no
more `work` Fields), a reduction's output is written as its element,
`[each N { body }]` builds an array element by element, and a missing
function's message names the used documents. Left: loop counts are
literals (`until index() >= n` is the pattern), and a scatter's target
is a Field because it holds what the sums start from.

The record showed one thing an agent
had to know, a keyword (`scan`) could not name a node, and the grammar
took that in: a node's name and a source reference accept any word,
since the one follows `node` and precedes `:` and the other follows
`<-`, `<~` or `from`.

Where the platform stands: four modes on three backends, bit for bit
between the CPU and the JIT and to rounding on the GPU, held by the
feature matrix (`crates/matrix`, 19 features x 5 modes on both); a GPU
session runs its scans on the JIT; `orla trace` (calls followed),
`run --profile`, `orla help` (90 builtins, generated from the typing
table); three protocol bridges and the C ABI (`orla-ffi`); the machine
substrate proven by the CHIP-8 and the Thumb core, the emulator itself
out of scope. Rated 9.3/10 in the last assessment; the remaining gaps
are the ones below.

1. The language's data types, agreed order, one round each:
   a. Done: fixed-size arrays `[T; N]` (any element type, nested too)
      in struct fields, `let`s, arguments, results and Field elements;
      literals `[1, 2, 3]` and `[v; N]`; a literal index is an
      `extract` checked at compile time, a run-time index the new
      `subscript` op, clamped like `at`. Chosen shape: an array is a
      struct of N fields marked with its element (`StructDef::element`)
      and interned by the compiler, so layout, construction, extraction,
      the mixers, the scatter combines and the trace came with the
      struct code on all three backends; only `subscript` was new (CPU:
      a word copy at a clamped offset; GPU: `alias S = array<T, N>` and
      `v[clamp(i, 0, N - 1)]`, which naga takes on a `let` array; JIT:
      the array spilled to a stack slot, one load per word of the
      element). Up to 4096 elements; `Extract`'s index widened to u32
      for it. The CHIP-8 font and program and the Thumb program are
      arrays now; their replays are bit for bit what they were. Not
      done, on purpose: array params (nothing crosses the ABI), an
      out-of-range failure (clamping is the Field rule), and hoisting a
      constant array out of the per-element code on the JIT (the ROMs
      run once at boot, so nothing asked for it; a scan reading a
      256-entry table every step would). Done since: the JIT keeps a
      constant table in read-only data and subscripts it in place, the
      interpreter folds it into one `Words` instruction (a four-million
      element, three-table Kernel: 524 to 40 ms on the JIT, 20 s to
      0.6 s on the interpreter); the marching and tetris pictures did
      not move, since their time is in `at` and the segment distances,
      not the tables.
   b. Done: matrices `Mat2/3/4` as ABI value types (column-major
      floats; `Type::order`/`column`), `mat3(c0, c1, c2)` and
      `mat3(diagonal)`, `m[i]` a column and `v[i]` a component (both
      through `extract`/`subscript`), `Mat * Mat`, `Mat * Vec`, `Vec *
      Mat`, scaling by a Float, `+ -`, `-m`, `==`; `transpose`,
      `determinant`, `inverse`. One formula set in
      `orla_rules::matrix` over an `Arith` trait: floats on the CPU, a
      Cranelift builder on the JIT (bit for bit with the CPU), text on
      the GPU for the `inverse_N` helpers (emitted only for the orders
      a Kernel inverts, since the unrolled 4x4 costs the shader
      compiler a quarter of a second). WGSL quirks met: naga has no
      unary minus on a matrix (`m * -1.0` is emitted), no `min`/`max`/
      `mix`/`==` on matrices (per column). The model example's orbit is
      one matrix. Not done: matrix params (nothing crosses the ABI as
      a matrix in a param; a Field of matrices works), `Mat / Float`
      (WGSL has none; `* (1.0 / f)`), non-square matrices, and
      `vec4(v3, 1.0)`-style constructors, which would be their own
      small round if a swizzle round ever comes.
   c. Done: integer vectors `IVec2/3` (`Type::int_components`,
      `lanes`, `other_kind`), `ivec2/3` from `Int`s, `icoord`, `isize`,
      `idomain`, componentwise `+ - * / % & | ^ << >> >>>` with each
      other or an `Int`, `-v`, `==`, members, `v[i]`, `abs`, `sign`,
      `min`, `max`, `clamp`, `float(v)` and `int(v)` between the kinds.
      The interpreter's integer instructions took a width and strides
      (the frame is read through `self` inside those loops, since a
      failure is reported through `self`); the JIT zips words; WGSL is
      native except that bit operators and shifts want both operands
      of one type (the `Int` side widens, a scalar shift count converts
      before it splats). Addresses: `at`, `put` and a scatter's pair
      take an `IVec` and the compiler converts it to the float address
      (`float_address`, and a synthesized `(Vec2, T)` pair struct for
      a scatter), which is exact for every coordinate a Field can hold,
      so no backend grew an integer Field path. The snake example's
      cells are integer vectors now. Not done: `IVec4` (no address
      needs it), integer `dot`, wrap arithmetic on vectors, `Bool`
      vectors, and integer vectors as params.
   Not planned: Bool vectors, unsigned ints (`ult`/`udiv`/`>>>` cover
   them), 64-bit, f16, strings, dynamic sizes, packed byte or RGBA8
   Field storage (a layout matter, not a type; f32 per channel is
   deliberate until memory or bandwidth says otherwise).
2. Decided against `at`/`at_linear` addressing modes, by the second
   play round (examples `scroll`, `blur`, `marching`): mirror is two
   lines of integer arithmetic on the cell (`blur.orla` `fold`,
   `mirror`) and wrap for `at` is `mod`; the one read that cannot be
   composed through `at_linear`, a wrapped linear read, is ten lines
   of four `at`s blended, now the picture package's `wrap.orla`
   `at_linear_wrapped`, which the scroll example uses and whose seam
   the replay pins against the clamped read. A mode would touch the
   IR and three backends' linear paths to save those ten lines; not
   worth it until a scene reads wrapped and linear in its inner loop
   and the four reads show in a profile.

The play round after the three types (examples `ant`, `cube`,
`tetris`, each replayed on the three backends) turned up, and fixed:
`size(port)` in a domain expression took the source node's domain
rather than the port's, so a scan's grid memory sized a consumer as
rows (evaluate.rs `of_source`); a replay `expect` decoded an `IVec`
field as floats (replay.rs); and Metal miscompiled a signed division
of a sixteen-bit-narrowed dividend (`(x >>> 16) % 7` gave 8192 for
zero; `udiv`/`urem` too), now guarded by `whole_*` in every integer
division and pinned by the matrix's `narrow division` row. The language
friction it noted was then acted on: `vec2/3/4` and `ivec2/3` take
shorter vectors among their components (`vec4(v3, 1.0)`), `.xy`,
`.zyx`, `.rgb` and the like are swizzles (both lowered to `extract`
and `construct`, so every backend had them; the matrix's `swizzles
and joins` row runs them), a function may take a math builtin's name
and stands for it (the Field, memory, context and construction
builtins stay reserved, `RESERVED_BUILTINS` in lower.rs), and an
`init` reading a memory with `at` is documented. The graph `domain`
point was a misdiagnosis: `size(cells) * zoom` takes an `Int` param;
the ant's `zoom` was a `Float` only because its Kernel declared it
so, and is an `Int` now. The tetris board clears lines with a
20-iteration loop calling a 10-cell `full` test per cell only in the
frame a piece lands, which the `if` keeps off every other frame.
3. What a session of play turns up: the voxel editor and the showcase
   were rated on interactivity; the JIT closed the CPU gap, and a GPU
   session now hands its scans to the JIT, so a machine runs as native
   code under either backend.

The recipe for a language addition, as this round settled it: IR
(`Intrinsic`/`Op`/`Type` with typing rule, `describe`, argument names),
compiler lowering (or composition into existing ops, which the backends
then get for free), the three backends, a `help` entry when not
generated, a DOCUMENTS line, a feature row in `orla_matrix::features`
(the matrix runs it under every mode on the JIT bit for bit and on the
GPU to rounding and lists every case that differs), then the full
verification. Declined this round with reasons recorded: a wide-lane
interpreter (the JIT covers it), the GDExtension (small gain, heavy
dependency), a Linux run.

Control flow was asked what it was missing, and two answers were
fixed. `&&` and `||` did not short-circuit: `n != 0 && 100 / n > 3`
divided, which the interpreter and the JIT called a failure and the GPU
did not, so they lower to an `if` and the optimizer writes the cheap
ones back to one operation before it hoists (the other side at most
four instructions, pure, unable to fail); alternating two binaries over
the replay suite showed no cost. And an `if` with no `else`, among a
block's statements, now guards what follows, desugared in the parser,
so a Kernel reads like the check it is. Left undone deliberately:
break with a value, which `until` covers, and a `match` on anything
but an `Int`.

The feature matrix then grew a second part. The features are what the
language can write; the semantics are what it promises. `not taken`
puts a division by zero on every side that must not run (an `if`'s
other arm, a `match`'s other arm, a short-circuit's other side, a loop
stopped by its `until`, a loop of no iterations), so a run that
finishes is the proof, and breaking one makes all five modes report
it. `edges` reads past every edge there is and holds the backends to
the same clamp; `signs` pins truncation toward zero and the
dividend's sign; `float edges` pins infinity, not-a-number and that
nothing orders against one. Writing them found a second float
divergence to document beside `inf * 0`: `sqrt` of a negative need not
be a not-a-number on a GPU under fast math.

Two forms the language was missing are in: a `match` arm takes a range
of values, `0..8` up to but not including its end and `0..=7`
including it, written out at parse time so the arms stay a table and a
value in two arms is still refused; and a `let` takes the fields of one
value, `let { x, y: top } = window();`, the value read once and each
field bound, also in the parser. Neither is known to the IR or to a
backend. The examples use both (the snake and voxel window and board
extents, the rubik hit, the shared text library's printable range) and
their replays are bit for bit what they were.

Three more the language was asked for, in one round. A function may
name type parameters, `fn twice<T>(v: T) -> T`, bound by the types a
call gives them and compiled once per set of them, on the machinery a
Field parameter already used; no example has needed one yet, so the
matrix carries them. A `break` among a loop body's statements stops the
loop after that iteration and gives that value: the compiler's work
alone, since a body that breaks carries the value and whether it
stopped, and the exit test reads the second. And a call to a function
that is not above the caller says why, rather than that the name is
unknown: a call to itself says a Kernel has no stack and a `loop` is
the form. What a `break` costs is the one carried word: the showcase's
ray march measured about one percent slower with it at 960x540, so the
hot loops keep their `until`; the chip8 key wait, which is a search,
reads as one now.

A round of play against the new language: `collapse`, wave function
collapse on a grid of the sixteen pipe tiles, every cell a bitmask of
what it could still be in a `Grid<Int>`. A reduce picks the cell with
the fewest left, a map settles it on one of them, and a map placed six
times spreads what that rules out; eight of those a frame, the grid
carried through `<~`, and the pointer melts what is under it. Every
step is integer work, so the replay pins its counts with no tolerance
at all on the three backends.

Writing it turned up three faults, all fixed. The GPU named a loop's
counter `i` and the loop's value number, which at thirty-two spells
`i32`: every `i32(...)` in that loop's body then parsed as a call of a
variable and the shader was rejected, so only a Kernel whose loop
landed on that number ever showed it (`nothing_the_shader_declares_
spells_a_type` writes the Kernel at every length so one of them does).
A `--save` split `node.port` at the first dot, so no node of a placed
graph could be named at all, and it now splits at the last. And a
`--save` may name one of the scene's outputs, which stands for what it
is wired to, while a name that is a placed graph says so rather than
saying it has no value.

A manifest says what its ports hold by name. A `Manifest` alone is
structural, as the ABI is, and prints a struct as its number; printed
beside a module's struct table (`manifest.named(&module.structs)`,
`orla_ir::type_name`) it says `Rows<Note>`, which is what `orla check`
and the protocol's `kernels`, `kernel` and `prepare` now answer with.
A layout that does not match is described in words too: "expects an
element of one float (f) but is bound to a Field of one integer (i)"
rather than "laid out as f ... i".

`orla bench` measures a scene the way this project has been measuring
one all along, and by hand every time: a warm-up frame thrown away, a
count of frames run several times, the least of the runs reported, and
the clock advanced by the frame count so the work is the same work
every time. `--backend all` puts the three in one table against the
first of them, `--nodes` adds what each node cost with the device
waited for (a node placed several times as one line), and `--json` and
`--compare` are the A and the B: the second run prints what the first
said and how far it is from it. A scene that does not change with the
clock reuses everything after the first frame, and the table says that
rather than reporting the time it took to find out.

The first thing `bench` measured was itself, and it was wrong: a GPU
dispatch is submitted and not waited for, so a frame timed around
`evaluate` was the asking and not the work (showcase read 0.36 ms a
frame at 960x540). A frame now ends by waiting for the device
(`Backend::settle_device`, the GPU's poll), which is what a Player's
frame does by presenting, and `run`'s own "evaluated in" waits too.
The numbers that came out are the ones to keep: showcase 5.9 ms a frame
on the GPU against 569 on the JIT, voxel 1.8 against 83, fluid 3.4
against 39, raymarch 0.6 against 11.

Then two ways to make the GPU's many small nodes cheaper, one declined
by measurement and one kept. A frame's dispatches in one submission
instead of one each: tried, and worse, since submitting each as it is
recorded lets the device work while the CPU records the next
(`collapse` 7.7 to 9.9 ms a frame, `fluid` 3.7 to 4.5, the others
unchanged); the per-submit cost is 60 microseconds and it buys that
overlap. Writing the context and the parameters into buffers kept from
the last node instead of making two a node: kept, `collapse` 7.64 to
7.50 ms over four alternating runs, nothing else moved, and a node of
a busy scene does two fewer allocations.

The measuring found its own work to do. The `collapse` example asks
what the sixteen tiles have on a side, and it asks with a loop over the
tiles, twice a neighbour, four neighbours a cell: on the interpreter
that was four fifths of a frame. The optimizer now runs a loop whose
count and initial value it knows and whose body only computes, so what
comes out is the literal it was always going to be (rewrite 8, in
`Simplify::run_loop`, with a budget of twenty thousand operations a
function). Nothing that reads a Field, writes a memory, calls a
function or can fail is run there: a failure belongs to the element
that has it. `collapse` on the interpreter went from 237 to 43
milliseconds a frame and the whole replay suite on the interpreter from
29.2 to 12.5 seconds; the other examples are unchanged, since none of
them happened to write a loop over constants, and the JIT and the GPU
are unchanged too. What it buys is that the clear way to write a table
is now the cheap one.

The protocol has the measurement too: `bench` takes `frames`, `repeat`,
`fps` and `nodes` and answers with the least of the runs, the median,
and what each node cost, the same numbers `orla bench` prints. The
clock advances while it measures, as `advance` advances it, so an agent
that edits a Kernel can ask what the edit cost and hold it against what
it asked before.

A round of play in the domain that was thinnest: `strings`, six plucked
strings on the keys `1` to `6`. A string is a delay line, a `memory`
carried frame to frame with `<~`, and a scan walks the block of 800
samples writing each slot back as the average of itself and the next,
a little quieter: Karplus-Strong, four lines of Kernel. A key that has
just gone down fills its line with noise at the block's first sample,
and nothing reads what that loop gives, which the optimizer keeps
because it holds writes. The picture is the lines themselves, each
drawn at its own length. The rendered sound measures 110.3 and 131.1 Hz
for the first two strings, which are the notes asked for, and the
replay pins samples and line slots that the interpreter, the JIT and
the GPU all agree on to a ten-thousandth. A frame is 22.9 ms on the
interpreter, 2.3 on the JIT, 0.9 on the GPU at 640x400. Nothing had to
be fixed to write it, which is the first time that has happened.

Then the GPU's own side of a frame, which for a scene of many small
nodes is mostly the CPU asking. A probe put a node's cost at about 90
microseconds before any of its work: twenty to turn an encoder into a
command buffer, thirteen to send it, twelve to write the two uniforms,
two in the output buffer, one in the bind group.

What was kept is the uniforms. A scene of many small nodes runs them
under the same context with the same parameters, so the bytes are
compared with what those buffers were last given and a node whose
uniforms have not changed writes nothing: twelve microseconds a node
became under four. A frame at 960x540, least of four runs of twenty
frames: `fluid` 4.40 to 3.34 ms, `collapse` 8.99 to 7.99, and
`voxel`, `million`, `showcase`, `raymarch`, `rubik` and `strings`
within a few percent of where they were, which is this machine's
noise. The memory copies a scan's ports need also go into the node's
own encoder now rather than one of their own, which is one submission
fewer for those.

What was tried and taken back out was sharing a command encoder
between small nodes. It was worth twenty percent on `collapse`, the
one scene in the suite with sixty-five tiny nodes, and nothing anywhere
else; and it asked for an invariant with a silent failure: a write
through the queue lands before anything submitted after it, so a
uniform buffer two nodes share cannot be written while the first
node's commands are only recorded. Breaking it gave wrong pixels in
every GPU replay and no error at all. A scene of many small nodes is
better served by having fewer nodes.

The same question was then put to the round before it: what did the
CPU side take on that fails quietly? Two things, and both are answered.

A Field made to be filled is given the words a released Field gave
back, and the caller says it writes every one of them. A caller that
left some would have shown another Field's contents and nothing would
have said so, so in a build with debug assertions those words are
filled with a value no Field ever holds: a partial fill is then a test
that fails rather than one that passes on what was lying there. The
whole suite runs in such a build.

The pile of released buffers was bounded by how many there were and
not by how large, so a scene of large Fields could leave the system
holding a hundred megabytes of what nothing was using. It holds a
handful of buffers and no more than an eighth of the Host's budget
now, and a buffer larger than that goes straight back.

One thing the JIT's fast path rests on is written down rather than
changed: the views it reads inputs through are made before the output
Field is, and they hold pointers into the inputs' words, which stay
good because a Field's words are an allocation of its own. A Host that
put every Field's words in one place would break that silently, so the
comment says so where the code is.

The rest of the CPU batch has nothing of that shape. The element split
and the ranged entry are arithmetic the replays check word for word;
the inlined reads clamp where the helpers clamped, which the matrix's
`edges` row holds them to on every backend; and a failure still stops
its run and names its own element.

Left as is: the interpreter reports an arithmetic failure in a scan at
`(index, 0)` and so does the JIT; the JIT's parity is against the CPU's
words, so a change to the interpreter's rounding needs a change here.

A scene of many small nodes was then answered at the root: a node
placed several times over its own output is one execution that
repeats. `times = N` with a `carry` writes N nodes, each taking the one
before it on one port and alike in every other way, so when nothing
else reads what passes between them the planner joins them into one
group that keeps the last node's name and the first one's inputs. A
backend that can repeat itself is asked first: the GPU records the
passes into one command encoder with two buffers taking turns, so
eight spreads are one submission and one set of uniforms rather than
eight; a backend that cannot falls back to running the group N times
and releasing what is between, which is what the interpreter and the
JIT do. The gate is narrow on purpose — the same Kernel of the same
generation, the same parameters and bindings, the same domain, Map
only, neither node async nor pinned, nothing else reading the middle —
and `orla graph` prints the run it made
(`steps/8.spread/6 + ... : collapse.spread@1`). Measured at 960x540,
least of three runs of sixty frames: `collapse` 7.35 to 6.47 ms on the
GPU and `fluid` 3.01 to 2.18; `voxel`, `million` and the rest where
they were, and the JIT within this machine's noise, since what a
repeat saves there is a dispatch and not the work.

Where the Kernel only `sample`s its input, ordinary fusion still gets
there first and inlines all N times into one Kernel, which is better
than repeating it. Repetition is what the ones fusion refuses get: a
Kernel that reads its neighbours, which is what `collapse` and `fluid`
are made of.

A scene someone else wrote found the largest single cost this project
had: a mesh editor of three nodes, a `struct Studio` holding two
`Mesh`es of `[Vec3; 96]` and `[[Int; 4]; 48]`, drawn by a ray-traced
Kernel over the window. It ran at 44 frames a second and the Player's
own numbers said the frame was mostly `present`, which is where a
GPU's work is waited for. The frame was 16.7 ms of GPU work for six
faces of ray tracing, and a probe of two fields of that element and
nothing else took 14 ms of it: every pixel was copying 481 words to
read six of them, a gigabyte a frame, which is the memory system and
not the tracing.

So a field, a column or an element taken out of a whole element of an
input is now a read of those words alone (`Op::Part`). The instruction
holds the way from the element to the part, field by field and index
by index, and each backend walks it: the interpreter one copy at an
offset it works out, the JIT a load, the GPU an indexed read of the
storage buffer. The path is in the instruction rather than in
arithmetic beside it because the interpreter dispatches instructions
and six of them per read cost it more than the copy did; written this
way it is one, and validation walks the path to the part's own type
rather than trusting a width. The whole read stays where something
wants the whole; a name is not such a want, so the `let` goes with the
read it named, which the compiled backends were already dropping and
the interpreter was not.

Reading only what is used is worth nothing while the words are copied
into a function's argument, which is what a `fn trace(m: Mesh)` does at
every element, so a function taking a record or an array of 32 words or
more is now written out where it is called however long it is, up to
1024 instructions. Inlined, its reads narrow like any others.

Measured on the same machine, one after the other, a frame of that
scene: 16.7 to 1.5 ms on the GPU and 168 to 11.5 on the JIT. The probe
of two fields: 9.25 to 0.70 on the GPU, 18.6 to 0.37 on the JIT, 13.3
to 10.3 on the interpreter. The examples are where they were, within
this machine's drift (which is a few percent over an hour of
benchmarking, and was worth finding out: two readings taken an hour
apart are not an A and a B). The scene's own interpreter frame is four
percent slower, since a part is read where a copy was made once and
read from the frame after: that is the trade, and the interpreter is
the reference rather than the fast one.

What the whole round also says about `run --profile`: it waits for the
device after each node and runs one cold frame, so on the GPU it
counts a submission and a round trip per node. It said the one-element
state node of that scene cost 11 ms while the whole frame is 1.5.
Read it for which node is heavy, not for what a frame costs; `bench`
answers that.

The Player's line had the same hole in it: a GPU's work lands in
`present`, where the display's refresh also is, so 22 ms of work and
16 ms of waiting for vsync read the same. With the line up it now
waits for the device once evaluating has handed it everything and
shows what that took as `device`, which leaves `present` the refresh it
is really waiting for. `device` and not `gpu`, which the same line
already says further along as the name of the backend: the first
spelling of it said `gpu` twice and meant two things. Asking costs the overlap between one frame's
device work and the next frame's own, so nothing but the line asks,
and a backend with no device of its own has no such field.

Then the half of that the round left: `sample`. A part of an element
read at this element's own place is the same narrowing, and the same
instruction with no address. What held it up was fusion, which replaces
a sample by the value the producer computed: a part of a value is not
a read at all, so fusion writes it out as the extracts and subscripts
the path stands for, walked against the producer's element type in the
fused module's own numbering. The JIT reads a sampled part from the
place the run already carries for that input, so it is the loads and
an offset and nothing else.

Measured, a grid of a 97-word element sampled for two of its words at
480x270: 1.29 to 0.71 ms on the GPU and 1.02 to 0.67 on the JIT, and
2.45 to 2.72 on the interpreter, which pays the same nine percent it
paid for `at` and for the same reason. The examples do not move: their
elements are a few words, so there was little to save; seventeen of
them read a sampled part now and their replays are bit for bit what
they were. A per-element cache of where this element sits in each
input was tried for the interpreter and taken out again: resetting it
cost more than the arithmetic it saved.

The other half asked about was local aggregates: an array a Kernel
builds and a record a function gives back, both made and copied whole.
Looking found one thing worth changing and one worth declining, both
measured. The JIT put an array down in a stack slot of its own for
every index of it, so a table read at three places stored its
thirty-two words three times; it stores them once per block now and
the later reads are loads, which took a table of thirty-two built and
read three times per pixel at 960x540 from 11.7 to 10.4 ms. The
examples do not move: the arrays they index are tables known when the
Kernel compiles, which already lived in read-only data.

Declined, with the numbers: writing out a function that gives back a
wide value, as one taking a wide argument already is. A function
building a 64-float table, called once per pixel, went 40.1 to 39.2 ms
on the JIT and 1.08 to 1.31 on the GPU, which is the shader growing
for nothing. A function that builds a wide value pays for building it,
not for handing it over, and the argument rule is what it is because a
copy at every element is the whole of that call.

A pad's analog values are in the context now, which the gamepad file
had said for months they belonged: `stick(0)` the left stick, `stick(1)`
the right, `stick(2)` the two triggers, and the pad's buttons as
`button(8)` upwards beside the pointer's three. One new operation for
all of it rather than three (a pair by number, as `button` takes a
number), and the existing `button` widened, which is why this cost the
IR one variant and each backend one arm: the interpreter reads the
pair, the JIT indexes three pairs in its context struct with the number
clamped and selects zero outside them, the GPU carries them in two more
words of the context head. The Player fills them from gilrs with a dead
middle taken out, and still presses the arrow keys from the left stick,
so every scene written before a pad existed plays on one.

A replay drives a pad (`stick left 0.5 -0.25`, `button south down`) and
the protocol takes `"sticks"`, so what a pad does is testable without
one; the matrix's `context` row now reads both sticks, the triggers, a
stick that does not exist and four pad buttons, under every mode, bit
for bit on the JIT and within rounding on the GPU.

This is the first half of what a scene someone walks through needs. The
second is a renderer that takes its geometry from Blender rather than
from a function: the pieces are already here (`native.rasterize@1` is a
visibility buffer, `orla import` writes the rows a scene reads, and the
bridge's Push writes those same rows with the Host watching them), and
what they are worth is 590 thousand triangles in 12.7 ms at 1920x1080,
where 37 thousand cost 11 — the frame is the pixels, and the triangles
are nearly free.

The Host's budget was a number nobody had chosen for anything: sixty-four
mebibytes of words, which is a fine net for a scene of pictures and a
wall for a scene of geometry. Ten million triangles asks for about four
hundred and sixty megabytes of Fields — the vertices, the same vertices
projected, and the indices — so it did not run, and what stopped it was
a constant rather than the machine or the device. The default is two
gibibytes now, and `--memory <MB>` states the policy where a scene wants
more: `run`, `render`, `replay`, `serve`, `graph`, `trace`, `bench` and
the Player all take it. A budget is still a net, and the message still
says what was asked for and what was left.

What ten million triangles cost, once they fit: 27.8 ms a frame at
1920x1080 on the GPU, against 11.1 for half a million and 18.0 for
three million — a fixed 9.8 ms of pixels and about 2.6 ms a million
triangles, with no culling of any kind and every triangle on screen.
The rasterizer itself is 2.4 ms of that; the rest is projecting every
vertex and shading every pixel.

That last paragraph used to end by saying a frame that culled what it
cannot see would carry several times the geometry at the same cost. It
was a guess, and measuring it says otherwise. A Kernel can cull: a
triangle whose three corners are one vertex is skipped, so a map over
the triangles writes `Tri { a: 0, b: 0, c: 0 }` for the ones it does
not want and the row keeps the length its domain fixed. It is correct —
the picture is bit for bit what it is with the triangles simply absent,
on all three backends, with the camera inside the mesh. What it is
worth is small. Two million triangles at 1920x1080: 5.53 ms as they
are, 5.52 with a culling node that keeps every one of them, 4.85 with
ninety-nine hundredths culled, 4.84 with all of them culled. A sheet
that really has the twenty thousand triangles the ninety-nine per cent
leaves is 2.57 ms.

So culling by collapsing a triangle removes 0.68 ms of two million —
the rasterizer's own per-triangle work, which was already the cheapest
part — and cannot touch the 2.27 ms that is projecting a million
vertices, because that Kernel's domain is fixed and it runs over every
vertex whether or not the triangle it belongs to survives. The lesson
is about the shape of the language, not the backend: culling in Orla
shortens no row. What a scene of ten million triangles wants is fewer
vertices to project, and the only thing that gives it is a graph that
places less geometry — not a Kernel that hides some of what it is
given.

Two real faults came out of measuring it, both of which any renderer
would have hit. Routing the triangle row through a Kernel at all —
even one that returns what it was given — made the fused module
invalid: `%36 branches yield S1 and S4`. Fusion appended the producer's
structs to the consumer's table, so the consumer's sample of the fused
input arrived carrying the producer's copy of `Tri` while the literal
in the other arm of the `if` carried its own; the two are the same type
and the validator could only see two indices. Structs that are the same
structurally are now one entry in the fused table, which is also what
makes the splice type at all. And the GPU did not keep the contract at
the near plane: `native.rasterize` skips a triangle with any vertex at
`w <= 0` whole, which the interpreter and the JIT do, and the device
clipped it and drew what was left — 330 pixels where the reference drew
none. Its vertex stage now reads all three corners' `w` and sends the
triangle to a single point when any of them is behind the eye, which
costs nothing measurable (two million triangles 5.54 to 5.51 ms, every
example within noise). `examples/mesh/tests/near.replay` is the whole
contract written down: a corner in front, the same corner at half the
`w`, at zero, behind, three corners collapsed onto one, and a corner
that is not a number — the picture the same on all three backends in
every case, and the old GPU fails it.

Two float divergences seen while doing this and left alone, both of
them the shader compiler's and not the rasterizer's: a literal
`1.0 / 0.0` in a Kernel compiles on the interpreter and the JIT and is
refused by the device at publish ("Divide operation overflowed"), and
`0.0 * x` where `x` is not a number is zero on the device and not a
number on the reference.

So the question became whether a graph can place less geometry at all,
and the answer was no: `domain` resolved before anything ran — numbers,
`count(port)`, `size(port)`, `window()`, `scale()`, arithmetic and
parameters — a side had to be positive so a node could not shrink to
nothing, `times` was at least one, and a parameter was a literal or a
parent graph's, so nothing a Kernel computed could size anything. The
only lever was a parameter from outside, which is cheap (swinging a
node between 1.18 million and 294 thousand triangles every frame costs
the average of the two and a tenth of a millisecond more, because the
Fields come from the pools) but puts the decision in the host.

Before building the closed loop, what the loop would be worth. A Kernel
can already read a count out of a Field and return early, which needs
no engine at all. On two million triangles at 1920x1080 that took the
frame from 5.50 ms to 4.81 with everything culled — it saves the
arithmetic and nothing else, because the element is still written. The
cost of geometry here is the Field: a million vertices of nine words is
thirty-eight megabytes a frame, and what is written is what is paid
for. That ruled out the mechanism this round was going to be — an
indirect dispatch, where the domain stays at its maximum and the
backend runs fewer elements — because the tail then has to be cleared
to stay deterministic, and a clear is still a write of the whole Field.
Only a shorter Field is shorter.

So `domain` may now read a number a Kernel wrote: `value(port)`, or
`value(port, i)` for element `i`. It is the one term that is data
rather than shape. `examples/decided` flies a camera over a sheet whose
`domain = value(detail) * value(detail)`, where `detail` is a Kernel
turning the camera's height into a resolution: 24 by 24 vertices high
up, 263 by 263 on the deck, a hundred times the geometry, with nothing
in the scene rewritten and no parameter carrying the number. On the
two-million-triangle sheet: 5.50 ms static, 5.69 with everything live,
4.30 at a half, 2.99 at a tenth, 2.68 at a hundredth — against the 4.81
floor a Kernel's own early return could reach.

No backend was touched. The number is read on the Host, when the node
binds, after its producer has run, so it is this frame's; on a device
that is a wait, about two tenths of a millisecond, once a frame however
many nodes read the same Field. That is what holds the port to a row of
scalars and to a thousand of them: it carries counts, not data, and the
wait can never turn into a hidden download of a scene's geometry. A
count that can reach zero wants `max(value(port), 1)`, since a domain's
sides must be positive, and the message says so where it happens.

One thing had to be taught: a node whose size a Kernel decides is not
knowable before its producer runs, so it fell out of fusion, which is
worth 0.28 ms on that sheet. A consumer with no `domain` of its own
runs over its first input's extent, so when that input is the producer
the two are the same extent whatever it turns out to be; the producer's
expression moves onto the fused node under the names fusion gives its
ports (`value(mesh/detail) * value(mesh/detail)`), and the chain folds
as far as it used to.

A replay can say how long a Field came out: `expect count
ground.result = 576`. Without it the capability could be measured and
not written down.

A Kernel could then say how many, and the next thing it wanted was to
say which ones. That needed no engine at all. A rank is a prefix sum
over the flags, which `scan` is; the count is a reduce; and putting
each kept index where its rank says is a scatter whose out-of-range
addresses are dropped, which is what a flag of zero sends. The three
are `packages/compact.orlapkg`: a row of flags in, the kept indices and
their count out. The prefix sum is ordered work, one thread on every
backend, and it costs 0.26 ms for four thousand flags on the device and
0.54 for sixty-five thousand — so it is a rank over the things a scene
decides about, chunks and bodies and lights, never over the elements
they expand into. A parallel scan is log N passes of dispatch overhead
and would lose at these sizes; it is worth having only past a few
hundred thousand, and nothing wants that yet.

`examples/culled` is the whole thing standing up: a world of `cols` by
`cols` chunks, a Kernel that flags the ones in front of the eye, the
compact package, and every geometry node at `domain = max(value(count),
1) * <per chunk>`. At 1920x1080 a world of ten million triangles costs
12.0 ms a frame against 26.1 for placing all of it; forty million costs
13.1 against 73.0; a hundred and sixty million costs 13.6, and placing
all of that does not fit in memory. Sixteen times the world for a
millisecond and a half — the cull test and its prefix sum over sixteen
times the chunks — because what a frame costs is what it can see and
the world above that is free. That is the answer to a question from
several rounds back: a scene carries geometry it could never draw, and
the way it does is that the geometry it cannot draw is never a Field.

The replay is the claim rather than the numbers: the same frame with
the culling on and off, `--param culling=false`, is the same picture to
the last pixel, from a third of the geometry. A cull that changes a
pixel is a bug, and this one would say so.

Two things that cost an hour each and are worth writing down. The
cull's side planes must cover the projection's: the test used a half
angle narrower than `aspect / f`, so it cut inside the picture and left
a sawtooth of chunk edges down both sides of the screen. And a fog
driven by the depth buffer's number is not a fog driven by distance —
`s.depth * 1.6` saturated at every pixel and the whole world came out
sky. The surface carries the world position in its `uv`, so the
distance is there to be had; a depth buffer's number is not one.

Then the measurement itself, because every round's conclusions had come
from comparing whole frames and the per-node numbers had never been
trustworthy. They were worse than untrustworthy. On the culled scene at
1920x1080 the profile said the rasterizer took 0.087 ms of a 9.0 ms
frame — the cheapest node in it — and the shading took 4.76. The truth
is 3.92 and 1.33: the rasterizer is the most expensive thing in the
frame and was reported as the least.

Three faults, each of which alone would do it. A native operation was
never measured: `Runtime::execute` set `last_measured` in the Kernel
branch and not the native one, so a rasterize was given whatever the
node before it had left standing, and its own submission was never
waited for, so the work it had queued was billed to the node after it —
which is where the shading's 4.76 came from. The runtime clears the
number before every execution now and asks after a native operation
too, and the rasterizer waits under a measured run and carries
timestamps across its two passes like any other pass. A backend that
did not time a call answers `None` and the caller's clock is used;
`crates/runtime/tests/mixed.rs` holds a backend whose clock answers for
Kernels and not for native operations, and it fails against the old
runtime.

The second: the device's own clock does not work here. The adapter
reports `TIMESTAMP_QUERY` and Metal through wgpu 30 resolves the query
set to zeros, every time — the code already knew and fell back to the
wait, but it asked again for every node of every frame, and asking is a
buffer mapped and a queue polled. It is asked once now and remembered
(`clock::UNTRIED`, `ANSWERED`, `SILENT`), and nothing prints a number
as the device's when it is the call's.

The third: what measuring costs was inside what was measured. One empty
submission polled to completion is 0.014 ms on this machine, and every
node carried it. `Backend::measurement_floor` is that, measured as the
least of sixteen, zero where the caller's clock is the whole
measurement and zero again where the device's clock answers, and it
comes off every line a profile prints.

What makes the numbers checkable is that they now add up. On the JIT,
where there is no device at all, the mesh example's nodes come to 2.247
ms against a frame of 2.219. On the GPU the culled scene's come to 9.90
against 9.20, and the gap is named in the output rather than left to be
noticed: a node measured alone waits for its own work, so the setting
up of the next one no longer hides behind it. `run --profile` says as
well that it is the first frame, which compiles and allocates, and that
`bench --nodes` is the one that measures a scene already running.

Then the ruler was used, which is the point of having made it. The
first thing it pointed at: a reduction over four thousand integers cost
0.68 ms where a map over the same four thousand cost nothing
measurable. Sixteen kibibytes of work and two thirds of a millisecond.

The fold halved a level per pass — element `i` of the next level is
`merge(in[2i], in[2i + 1])` — so a level of `n` took log2(n)
dispatches, twelve for four thousand and twenty for a million, each
with a bind group and a uniform buffer of its own. A run of sixteen
folded at once is four of those levels, and the reason it may be done
at once is that it is the same tree: folding a run by the pairwise rule
reaches the same expressions as walking the levels one at a time, ragged
ends included, which `orla_rules::reduce`'s test states over counts from
one to four thousand. So nothing moved a bit. Every backend's reduction
of the same rows gives what it gave before, on counts that divide by
sixteen and counts that do not.

The first attempt did move something, and is worth writing down: folding
a run left to right rather than as a tree made the JIT three times
slower on a million elements (1.94 to 6.64 ms). Sixteen merges in a row
are a dependency chain and a level of pairs is sixteen chains a machine
can run at once. A tree within the run costs a device nothing and keeps
the machine's hands free, so the interpreter and the JIT still walk one
level at a time and did not change at all.

The second thing the ruler pointed at was the disease of two rounds
back, in the two places the cure had not reached: a reduction asked the
driver for its level buffer and a scatter for its target on every
execution. They come from the pile now. A reduction over a million went
0.71 to 0.54 ms with that, and a reduction over four thousand is 0.35
where it was 0.59.

What it is worth on real scenes, at 1280x720, least of three runs of
twenty frames: `collapse` 5.64 to 3.78 ms, `million` 2.31 to 1.72,
`spectrum` 0.90 to 0.81, `learn` 3.57 to 3.44, `culled` 7.37 to 7.11,
and nothing slower. The 123 replays are bit for bit throughout.

Two measurements that came out of the same pass and are worth keeping.
A node costs 0.027 ms in a frame, measured as the slope of one to
thirty-two tiny nodes (0.19 to 1.02 ms), which is what a scene of many
small nodes pays and what a profile's 0.19 ms per tiny node is not: the
rest of that is the serialization a measured run adds. And a scatter
carries about 0.19 ms over a map whatever its size, which is its two
passes — `last` applies with atomics and then gathers — and is
structural rather than wasteful.

What that per-node cost is, since it is the largest thing left that
every scene pays and an earlier round had declined to fix it on a
reason that turns out to be the wrong one. Measured as the slope of one
to a hundred and twenty-eight tiny nodes: the GPU 26 microseconds a
node, the JIT 2.4, the interpreter 4.5. So about 2 of it is the graph
and 24 is the device backend's.

It is not the submission. Timed inside `execute`: staging 3
microseconds, the bind group 0.9, the submission 8, the error scope
0.1 — 13 in all, and the earlier round's figure of 60 for a submission
is not what this machine does. Nor does the 13 show up in a frame,
because it is the CPU's and the device is working on the node before
while it happens. What the frame pays is the device's own launch of a
pass, which is most of the 26 and is not ours to spend or save.

The proof is fusion. A hundred and twenty-eight nodes in a chain, each
reading the one before at a computed address so none of them fuse, is
3.41 ms a frame; the same hundred and twenty-eight reading `sample`, so
they fuse into one, is 0.89. Fusion is the only thing that takes a pass
away, and it already does.

Batching the frame's commands into one encoder and one submission was
tried again, properly this time, and it is not merely not worth it: it
is unsound. The context and the parameters live in one pair of uniform
buffers kept between nodes and rewritten with `queue.write_buffer`,
which orders against submissions and not against recording — so every
node of a batch reads the last node's uniforms. Twenty-six of the
forty-one replays failed on the GPU, which is what that looks like. To
batch, every node would need uniform buffers of its own, trading an
allocation for a submission that was not the cost. And the shape of the
curve says so as well: at a batch of four the tiny-node probe went 3.54
to 3.33 ms and real scenes did not move, and at a batch of two hundred
and fifty-six it went to 4.65, because the device waits for the whole
frame to be recorded before it starts any of it. Declined for good, and
now for the right reason.

One thing the device could do and wgpu cannot ask for: several
dispatches in one compute pass, which is where Metal's per-encoder cost
goes. Consecutive dispatches in one wgpu pass have no barrier between
them, and a chain of nodes needs one, so every node is a pass of its
own. That is the floor until a backend can say "dispatch, then a
barrier, then dispatch" inside a pass.

Then the culled world was driven rather than looked at, which found the
thing looking at it could not. A hand steers it now — the left stick or
the arrow keys, and the automatic sweep when there is no hand, so the
replay is unchanged — and a replay drives a pad, so a roam is a test. A
thousand two hundred frames of turning leave the camera at (0.117,
0.245) in a world one and a half across: the wrap holds and nothing
drifts. In the hundred and sixty million triangle world, steering, the
kept count sits at 1419, 1422, 1421 of sixty-five thousand chunks
across the roam, and the window may be resized mid-flight.

What that turned up: the buffer pile matched a size exactly, and a
scene whose Fields change size never matches. `culled` keeps a
different number of chunks every frame — 1419 then 1422 — so every
geometry buffer it asked for was a size the pile had never seen, and
every one came from the driver. The pile takes the smallest buffer that
is big enough now, and never one more than twice the size. Nothing
reads past the element count the uniform gives — no shader asks a
buffer how long it is — so the tail of a bigger one is never touched.
`culled` at 1920x1080 went 8.61 to 6.93 ms, and the hundred and sixty
million triangle world 12.59 to 9.82. Every other example is unchanged,
because every other example has Fields of a fixed size; this is the
cost of the shape the last few rounds made possible, and it was hiding
where only driving the thing would find it.

The other half of the round was narrow and is worth stating as a rule
rather than as a change. A scatter's `last` cannot be one pass: an
invocation that has done its atomic does not know whether a later
source will beat it, and there is no barrier across a dispatch, so a
backend writes the winners in one pass and gathers their values in
another. Sixty-four bit atomics would pack the pair but still leave the
unpacking. What a scene can do instead is not ask for `last` when its
addresses are unique — `compact.gather`'s are, since a rank is unique —
and `max` over integers is one pass of atomics and nothing else: a
million sources 1.30 ms to 0.96, four thousand 0.41 to 0.35. The
package says why in its own source, since the reason is what makes it
safe.

The third thing the renderer wanted turned out not to be a change to
Orla at all. A texture running away from the eye crawls because
`at_linear` blends four texels of one Field and a distant pixel covers
many; the old answer is a mip chain, and the question was whether it
needed the Host to carry levels or `at_linear` to take one. It needs
neither. The strip of halvings is a sub-graph placed `levels` times with
`carry`, and the level a pixel asks for is the neighbouring pixel's
`uv`: a surface the rasterizer left carries `uv` everywhere, so the
difference between neighbours is exactly what a fragment shader's
derivative would say, and the triangle's `id` says whether the
neighbour is the same surface. It is a library —
the picture package's `mip.orla` and `mip.osn` — and the `mipmap` example
holds it: two pixels a little apart in the distance are the same grey
with the levels and are nearly black and nearly white without them, on
all three backends, which is the crawl written down as a replay.

One thing that cost an hour and is worth writing down: a node placed
`times` over does not know which repetition it is. A chain that has to
know (each pass writing the next level) carries the number beside the
data as a row of one `Int`, which the two-`carry` form allows. Without
it every pass rebuilt level one, which looked exactly like a chain that
was not being built at all.

The rasterizer has a second form: `native.rasterize@2`, which takes a
mask and a cutout. A fragment whose alpha in that mask is below the
cutout is not drawn — no surface, no depth — so what is behind a leaf's
edge is seen through it. Without it, foliage has to be geometry: every
needle a triangle, which is how the procedural tree of the round before
ended up costing 154 nanoseconds a pixel for one tree. With it a leaf
is two triangles and a texture, which is what every engine does.

The mask is an `Image` and its alpha is read where `at_linear` would
read it, by one formula both backends call
(`orla_rules::raster::mask_alpha`), so the cutout's edge is the same
pixel on the device and on the reference. The GPU discards in the
fragment shader; the reference tests before it settles depth, which is
the part that matters — a fragment that is not drawn must not hide what
is behind it. The parity test puts a checkerboard's holes over a near
quad and finds the far one through them on both.

What several scenes share stopped being an example. `examples/shared/`
had become a library with no way to be one: a scene of anyone's own
that wanted the mip levels would have written
`use ".../orla/examples/shared/mip.orla"`, with the word `examples` in
the path of the thing it depends on. The mechanism for this already
existed and was being used by exactly one directory, which was itself
inside `examples/`.

So there are packages now, in `packages/` beside the examples:
`picture` (wrapped reads, tone mapping and the display graph, mip
levels and the strip), `text` (the font, its baking, and the Kernels
that draw with it), and `pixelfx`, moved. Each has the manifest a
package has, and `orla package` describes all three. An example reaches
one the way anyone's scene would — by path, or by name where
`ORLA_PACKAGES` lists the directory — which also means the examples now
demonstrate the same thing twice: what they are for, and how a package
is used.

Moving them found a hole in the mechanism. A package could be reached
by name for its Kernels and only by path for its presets, so a package
used by name could not have its graphs placed — and a preset that was
found by name then looked for its own `use` beside where the scene said
it was rather than where the package is. One rule answers both: a path
that is not there and runs through a `<name>.orlapkg` is looked up
under `ORLA_PACKAGES` from that name on (`in_package`), and both the
document loader and the `use` expansion take it. A scene outside the
repository now places `picture.orlapkg/graphs/display.osn` with nothing
but the environment variable set, which is what the packaging was for.

Nothing else moved and nothing changed inside the files: the 114
replays pass unchanged, which is what a rename of this size is for.

What a full-screen pass costs on the GPU turned out to be mostly the
asking. A map node writing one image at 1920x1080 took 1.58 ms, which
is 0.65 nanoseconds a pixel for sixteen bytes — a twentieth of what the
memory can do — and the output buffer was made fresh every execution
and dropped with the Field that held it. A pile of released buffers,
taken by size and given back when the Field is released, puts that at
0.62 ms: 0.17 nanoseconds a pixel, which is the bandwidth. Two passes
went 3.20 to 1.37. The rasterizer's own copies, the same size every
frame, come from the same pile, and two million triangles at 1920x1080
went 13.0 to 9.2 ms.

Every example is faster for it, at 1920x1080, least of three runs of
twenty frames: `strings` 1.80 to 0.93, `model` 1.55 to 1.16, `million`
3.69 to 2.87, `raymarch` 0.50 to 0.39, `mesh` 0.71 to 0.59, `collapse`
7.29 to 6.10, `voxel` 7.61 to 6.65, `fluid` 2.17 to 1.94, and nothing
slower. It is the widest kind of win there is: every renderer anyone
writes over this backend pays that cost on every pass, and nothing
about the optimization has an opinion on what is being drawn.

What makes it safe is the queue's order. A buffer goes back to the pile
when the Field that held it is released, so nothing in the scene refers
to it, and whatever read it was submitted before whatever writes it
next; one queue runs its submissions in order. The pile holds twelve
buffers and half a gibibyte, which is what a few full-screen targets at
4K come to; past that they are the driver's again.

This is also a correction: the ten milliseconds that looked like a
shading pass in a `run --profile` was the per-node device wait, not the
pass. The measurement that matters is a frame.

The rasterizer had the same disease and worse. At 1920x1080 it cost the
same for ninety-eight triangles as for a hundred and thirty thousand —
5.6 ms either way — which says the cost was never the geometry: it was
2.5 nanoseconds a pixel, fifteen times what a map node's write costs,
for a surface of seven words.

Two things, both of them moving data that did not need moving. The
targets were copied into padded buffers and gathered from there, so the
surface was written, copied and gathered; the compact pass reads the
targets where they were written now, which wanted nothing but
`TEXTURE_BINDING` on them and took 5.6 to 4.0 ms. And the three targets
were made fresh every call — seventy-four megabytes of texture at
1920x1080 — so they come from a pile by shape and go back at the end of
the call, which took 4.0 to 2.7. Two million triangles at 1920x1080
went 9.3 to 5.5 ms, and the per-pixel cost of a surface is 0.99
nanoseconds where it was 2.5.

The pattern is the same one twice: what a pass costs is mostly what it
asks the driver for. A pile bounded by count and bytes, taken by shape
and given back when nothing holds it, is worth more than anything
clever about the work itself.

The interpreter had a third kind of the same disease, on the other
side: it was compiling the Kernel again on every execution. `prepare`
built nothing, so each `execute` walked the module and laid out every
value's words afresh — about seventy nanoseconds a statement, paid
per frame per node whatever the domain. A thousand-statement Kernel
over one element cost 80 microseconds a frame, of which 70 was the
compiling. The word code is a [`Code`] now, built once where the
backend prepares the module and held by the `Prepared` the runtime
already keeps, and that frame is 10 microseconds. A hundred-and-twenty
statement Kernel over sixteen elements went 90 to 60 microseconds, the
rest being the sines it actually computes. Nothing moved on the
examples, which are all pixel-bound and compile once per twenty
thousand elements; what this is for is a graph of many nodes over
small domains, which is what interactive logic looks like. The JIT was
checked for the same thing and does not have it: it compiles in
`prepare` already, and the same one-element Kernel costs it nothing
measurable.

A direct `CpuBackend::execute` still compiles on the spot, so the
crate stands alone; `execute_with` is the one that takes a prepared
[`Code`], and `trace` deliberately compiles its own, without the
optimizer, so it can still follow the calls.

## Open

A structure round against the Code Rules (no behaviour change; the
matrix and every replay bit for bit): the rules every backend shares
moved out of the CPU crate into `orla-rules`; `gpu::execute` is a
skeleton over `stage`, `fold`, `combine` and the adoption of outputs;
the interpreter is `eval/{code,compile,run}.rs`; the lowering is
`lower/{mod,declare,expr,builtins}.rs`; the matrix formulas' text form
builds `String`s, so nothing is leaked; and the runtime takes IR
(`prepare_module`), the session compiling what it publishes, so
`orla-runtime` no longer depends on the language crate. Left as they
are, deliberately: the flat dispatch matches (`Machine::step`,
`Fx::instruction`, `validate::instruction`, `Builder::value`,
`Builder::call`), each one arm per operation, which a split would
only scatter.

Holding that million against the old Orla (the one in
the previous implementation, whose benchmark is a million rows of
`vec2` moved by their own velocity, no picture) found something this
project had been carrying since the beginning: a map over `Rows` ran on
one thread. Both the interpreter and the JIT split their work by rows
of the domain, and a `Rows` domain is one row of a million, so nine
cores sat still. They split by elements now, which is the same work and
the same words. Measured: that benchmark's million rows went from 6.35
to 3.40 ms on the JIT and from 55.5 to 16.1 on the interpreter; the
`million` example from 20.4 to 12.7 and from 157 to 52.9; `learn` from
56.6 to 22.8; the interpreter's whole replay suite from 12.5 to 9.4
seconds. A grid was already split across cores, so pictures did not
move, and `particles`, which is mostly a scatter, did not either.

Then the rest of that gap, measured one piece at a time rather than
guessed at. A map's entry takes a range and walks it itself, so a
million elements is one call and not a million (`map_run`); `sample`
and an `Int` `at` are loads where they were calls into Rust through a
stack slot (`element_words`); a map with one output writes straight
into the Field it just created, where it used to fill a buffer of its
own, gather that into a second, and copy that into the Field; and the
Host keeps the words a released Field gave back, so the Field a node
makes every frame is the same memory every frame. What each was worth,
on the old benchmark's million rows: the range 3.40 to 3.15, the loads
3.15 to 2.84, writing in place 2.84 to 0.64, the spare buffers 0.64 to
0.58. The interpreter kept the copies it needs to stay safe but writes
its one output whole rather than gathering it first.

Then the addressing itself, which measurement said was worth three or
four times what a thread pool would be. A map's entry reads each
input's view once before its run rather than at every element, and
because a run stays on one row it can work out where that row begins
at the same time; then it carries every input's place from element to
element, walking on by one and stopping at the end of the row, which
is exactly what clamping an `x` past the edge comes to. A `sample` is
then the loads and nothing else: no extent, no clamp, no multiply. The
benchmark went 0.58 to 0.45 for reading the views once and 0.45 to
0.42 for carrying the places.

Where that leaves the comparison, the same work on the same machine: a
million rows of `vec2`, each moved by its own velocity, fed back, no
picture. The old Orla takes 0.270 ms on the CPU and 0.275 on the GPU;
this one takes 0.42 on the JIT and 0.70 on the GPU, having started at
6.35 and 0.63. At 0.42 ms the frame moves 24 MB, which is 57 GB/s, so
what is left is the memory system and the frame's own work (a graph
evaluation with a state commit, against a tick of one kernel) rather
than anything the code generator is still doing per element.

What the whole chase did to the examples, a frame at 960x540 on the
JIT: `learn` 56.6 to 7.7 ms, `million` 20.4 to 5.5, `fluid` 38.6 to
23.6, `voxel` 83.1 to 78.9, `raymarch` 10.7 to 10.3. The interpreter's
replay suite went from 12.5 to 9.1 seconds and the JIT's runs in 1.2.

A million of something, which is the number other engines are held to:
`million`, a million bodies each with a place, a speed and a tint,
carried frame to frame, pulled towards the middle by an inverse square
so a disc winds into rings, and drawn by a scatter of a million writes
into the cell each lands on. At 960x540 a frame is 2.4 ms on the GPU
and 20.4 on the JIT; four million is 9.3 and 73.1; a hundred thousand
is 0.84 and 4.4. Of the GPU's 2.4 ms the step is 1.5 and the picture
0.6.

Writing it turned up the sharpest measurement of the session. The
scatter summed a `Float`, and floats do not add the same way in every
order, so the GPU sorted a million pairs by their cell and added them
up in element order to give the interpreter's words: 24 ms of a 28 ms
frame, which made the GPU slower than the JIT. A count is an `Int`, and
integers do add the same way in every order, so the same scatter became
one atomic each and the frame became 2.4 ms. The picture is identical.
The rule a Kernel writes decides whether a million writers need an
order between them, and nothing else in the language says so as
plainly.

The two bridges with runners of their own are run by the suite now,
when the tools they need are on the machine: `bridges/web/test.mjs`
through node and a headless Chrome, and `bridges/godot/test.gd` through
Godot with `-s`, each given the test binary's own `orla`. Both pass
against everything this session changed (the protocol's new ops, the
named manifests, the device wait), which is what the round was for;
what is new is that the next change to break one breaks `cargo test`
rather than waiting for someone to open a bridge. A machine without
node, Chrome or Godot skips them and says so.

`orla new <dir>` writes a directory that runs: a Kernel, the scene, a
replay of what it draws, and a README saying what to change first.
Three kinds, because the three things people start with want different
shapes: `picture` is one Kernel over the window, `sound` a voice with a
drawing of it, `game` a walker the arrow keys move with two `<~`
bindings and a fading trail. The test makes each one, checks it and
replays it, so a change to the language that would break a template
breaks the suite instead of someone's first fifteen minutes.

Measured, not fixed, because there was nothing to fix: a session that
evaluates for a long time holds steady. Four thousand frames of
`collapse` at 640x400 left the resident size where it started on the
JIT (30 MB) and on the GPU (121 MB), and a session that publishes a
Kernel again and again, evaluating in step, keeps one version of it
(the test below pins that, and that nothing prepared is left
uncollected, and that the Host holds what one frame holds). What
republishing does cost is under a kilobyte a version that the allocator
keeps rather than returns: twenty thousand republishes grew the
resident size by 16 MB on the JIT and stopped growing when they
stopped. An agent that republishes once a second would carry a few
megabytes an hour of that, which is the allocator's business and not a
version's.

Seen once, then found: the protocol test whose served child's answer
went missing. `quit` on standard input ended the process from the
request loop while a thread of its own was still writing the answer to
it, so about one run in a hundred lost the last line. An answer to a
local request is written where it is made now (`protocol::Reply`), and
a connection's answers still go through its own writer, which is what
a socket needs. Two tests in that binary also shared a scratch
directory and deleted it under each other; each has its own now.

Not hand-tested: the Player's real right and middle buttons and wheel,
gamepad hardware (none connected), and a live resize of a window-sized
scene in the Player (the Player was opened on the voxel editor and
presented; the resize mechanism is covered by the graph test, not by a
hand on the window's corner). The web bridge's page was driven in a real
Chrome as well as headless: the text example's title republished from
its editor and the voxel editor filled a Retina canvas at 24 ms a frame
and took a click; a browser refuses `file://` pages through the Chrome
extension, so the directory was served with any static file server for
that (`python3 -m http.server`), which the page does not need otherwise.
The Godot bridge's window was opened (no script errors) but not
clicked through by hand; its demo scene is driven by the headless test
instead. The Blender bridge ran on Blender 5.2
only; Watch polls (default 0.5 s) and reads whole rows each tick. The file
watch polls every 200 ms and can observe a half-written file; the next write
is picked up. The CPU interpreter is the reference; it compiles to
word-addressed code and runs the raymarch scene at 400x260 in about 0.35 s
and a voxel frame at 960x540 in about 1.3 s on ten cores, so it is still
not interactive for those; the JIT runs the same frames in 22 ms and
92 ms. The worker is one FIFO thread.
Audio pacing is one block behind the ring.
