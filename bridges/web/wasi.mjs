// Runs a program Entelechy built for WebAssembly under WASI (a test binary,
// or the command line) in node, as the host of the code Entelechy's JIT
// writes: the same `entelechy.load` the page gives it (entelechy.js). Cargo uses it
// as the runner for `wasm32-wasip1`:
//
//   CARGO_TARGET_WASM32_WASIP1_RUNNER="node bridges/web/wasi.mjs" \
//     cargo test -p entelechy-jit --features wasm-host --target wasm32-wasip1
//
// The program sees this machine's files, and the current directory is
// its own; its arguments follow the file.

import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { WASI } from "node:wasi";

process.removeAllListeners("warning");
const here = dirname(fileURLToPath(import.meta.url));
await import(join(here, "entelechy.js"));
const [file, ...args] = process.argv.slice(2);
const wasi = new WASI({
  version: "preview1",
  args: [file, ...args],
  env: process.env,
  // The current directory, for relative paths, and the root, for the
  // absolute ones a test builds from where its crate is.
  preopens: { ".": process.cwd(), "/": "/" },
  returnOnExit: true,
});
let instance = null;
instance = new WebAssembly.Instance(new WebAssembly.Module(readFileSync(file)), {
  ...wasi.getImportObject(),
  entelechy: globalThis.Entelechy.host(() => instance),
});
process.exitCode = wasi.start(instance);
