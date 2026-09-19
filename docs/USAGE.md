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

Evaluate a `.osn` graph and write the output image as a binary PPM. Kernels are
resolved from `kernels/` next to the scene unless `--kernels <dir>` is given.
External images declared with `input <name>` in the scene are bound with
`--input <name>=<file.ppm>`.

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm
```

## Hot reload

Keep the scene running and republish Kernels as their files change:

```sh
cargo run -- run examples/grayscale/scene.osn --out output.ppm --watch
```

Edit and save any `.orla` file under `kernels/`. Only nodes affected by the
change re-execute; the output file is rewritten. A file that no longer compiles
is reported and the previous version keeps running. Removing a file retires its
Kernel. Editing the scene reloads the graph.

## Player

Open a scene in a window and keep it live. Kernel and scene files are watched
exactly as in `--watch`; Kernels that declare `time.read` re-render every
frame, pure nodes stay cached. Use a release build for interactive frame rates.

```sh
cargo run --release -p orla-player -- examples/material-ball/scene.osn
```

`--scale <n>` sets the initial window magnification and `--input` binds
external images as for `run`. Escape closes the window.

## Where the contracts live

- Language syntax: [the frontend](../crates/language/src/lib.rs) and its parser tests.
- Capabilities and types: [the ABI](../crates/abi/src/lib.rs).
- IR operations and validation rules: [the IR](../crates/ir/src/lib.rs) and
  [its validator](../crates/ir/src/validate.rs).
- Scene document format: [the graph document parser](../crates/graph/src/document.rs).
- Publication semantics: [the runtime](../crates/runtime/src/lib.rs).
