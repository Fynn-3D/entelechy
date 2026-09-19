# Usage

Orla is a Rust workspace with no external dependencies. Build and verify:

```sh
cargo build --workspace
cargo test --workspace
```

## Check a Kernel

Compile `.orla` files and print their manifests. Diagnostics carry
`file:line:column`. Add `--ir` to print the validated Orla IR.

```sh
cargo run -- check --ir examples/grayscale/kernels/grayscale.orla
```

## Run a scene

Evaluate a `.osn` graph and write the output image as a binary PPM. The
document names its Kernel sources with `use`; `--library <dir|file>` adds
more. External images declared with `input <name>: Image;` in the scene are
bound with `--input <name>=<file.ppm>`; graph parameters are overridden with
`--param <name>=<literal>`.

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm
```

## Inspecting a graph

```sh
cargo run -- graph examples/grayscale/scene.osn
```

prints the flattened graph in canonical form, the published Kernels with
their generations, and the plan: which groups execute, in order, with the
producers each one absorbed. In the Player, Tab cycles the preview through
the nodes (pinning each so fusion keeps it materialized) and back to the
output; the title shows the node in view, the frame time, what ran and what
was fused, and how the frame reached the window.

## Fusion

Chains of pointwise nodes run as one Kernel: a producer whose only reader
samples it at the current coordinate over the same domain is inlined into
that reader, and its image is never written. `run` reports fused groups as
`consumer+producer`. Pass `--no-fusion` to `run` to compare, or pin a node
through the protocol to observe its image.

## Backends

`run` and the Player take `--backend cpu|gpu`. The CPU interpreter is the
reference and reports per-pixel integer failures; the GPU backend runs the
same IR as WGSL compute through wgpu, keeps images resident on the device
between nodes and frames, and reads pixels back only when something
observes them. It is the one to use for anything interactive. The Player
defaults to the GPU when a device is available.

```sh
cargo run --release -- run examples/raymarch/scene.osn --backend gpu --out march.ppm
```

## Hot reload

Keep the scene running and republish Kernels as their files change:

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm --watch
```

Edit and save the scene or any document it uses. Only the Kernels whose text
changed republish and only the nodes they affect re-execute; the output file
is rewritten. A Kernel that no longer compiles is reported with its position
and the previous version keeps running. A Kernel that leaves its document
retires.

## Player

Open a scene in a window and keep it live. Kernel and scene files are watched
exactly as in `--watch`; Kernels that declare `time.read` re-render every
frame, pure nodes stay cached. Use a release build for interactive frame rates.

```sh
cargo run --release -p orla-player -- examples/material-ball/scene.osn
```

`--scale <n>` sets the initial window magnification, `--backend` picks the
backend, and `--input` binds external images as for `run`. On the GPU the
window is drawn from the resident output directly; the title shows the
presentation path and how many frames had to be downloaded. The pointer and
keys reach Kernels that declare `input.read`; F5 resets Graph state; Escape
closes the window.

## Replay tests

An application's tests are `.replay` files next to its scene: scripted frames,
pointer and key input, state resets, and pixel expectations. No Rust is
needed to test a scene.

```sh
cargo run -- replay examples/life/tests/blinker.replay
```

The file format is documented at the top of
[the replay module](../crates/cli/src/replay.rs). Key codes follow the common
browser codes: letters and digits by ASCII uppercase, space 32, enter 13,
arrows 37 to 40.

## Documents

[The document format](DOCUMENTS.md) defines `.orla` libraries and `.osn`
graphs: one grammar, wiring written at the consumer, inline Kernels, and
`use` for reuse.

## Creation protocol

Agents drive a session over stdin and stdout with JSON lines:

```sh
cargo run -- serve examples/life/scene.osn
```

See [the protocol](PROTOCOL.md) for the operations. `orla-player --serve`
runs the same protocol inside the window, so an agent edits what a person
is watching.

## Where the contracts live

- Language syntax: [the frontend](../crates/language/src/lib.rs) and its parser tests.
- Capabilities and types: [the ABI](../crates/abi/src/lib.rs).
- IR operations and validation rules: [the IR](../crates/ir/src/lib.rs) and
  [its validator](../crates/ir/src/validate.rs).
- Document format: [the contract](DOCUMENTS.md), [the frontend](../crates/language/src/lib.rs), and [graph lowering](../crates/graph/src/document.rs).
- Publication semantics: [the runtime](../crates/runtime/src/lib.rs).
