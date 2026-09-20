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
`examples/shared/text.orla` declares the glyph row and the drawing, and the
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
   of four `at`s blended, now `examples/shared/wrap.orla`
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

Left as is: the interpreter reports an arithmetic failure in a scan at
`(index, 0)` and so does the JIT; the JIT's parity is against the CPU's
words, so a change to the interpreter's rounding needs a change here.

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

Seen once, not again: `crates/cli/tests/protocol.rs`
`fields_come_in_through_write_and_buttons_reach_kernels` failed in a
full `cargo test --workspace` that ran beside a release replay loop:
the served child's stdout ended before its first answer (`serve`
defaults to the CPU backend, so not the GPU). It did not come back in
25 runs of that test binary under GPU test load nor in two full
suites under the replay loop. The harness now drains the child's
stderr on a thread and, when an answer is missing, reports the child's
exit status and everything it said, so a next time is a diagnosis
rather than a position.

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
Audio pacing is one block behind the ring. Native manifests print element
types as anonymous structs.
