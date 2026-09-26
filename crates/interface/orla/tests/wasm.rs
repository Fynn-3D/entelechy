//! Orla as WebAssembly, a machine whose `usize` is 32 bits and whose JIT
//! writes WebAssembly: the tests of the core, the front end and the
//! interpreter, and the JIT's own (the parity cases, the feature matrix,
//! random programs, each held to the interpreter word for word), built
//! for `wasm32-wasip1` and run in node, which hosts the JIT's code as a
//! page does (`bridges/web/wasi.mjs`). Built for release, as Orla runs in
//! a page: a debug build's frames are too large for the engine's own
//! stack at the deepest nesting the language allows. Skipped with a word
//! where this machine has no node or no such target.

use std::path::{Path, PathBuf};
use std::process::Command;

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn node() -> Option<PathBuf> {
    ["node", "/usr/local/bin/node", "/opt/homebrew/bin/node"]
        .into_iter()
        .map(PathBuf::from)
        .find(|path| {
            Command::new(path)
                .arg("--version")
                .output()
                .is_ok_and(|out| out.status.success())
        })
}

fn has_target(target: &str) -> bool {
    Command::new("rustc")
        .args(["--print", "sysroot"])
        .output()
        .ok()
        .map(|out| PathBuf::from(String::from_utf8_lossy(&out.stdout).trim()))
        .is_some_and(|sysroot| sysroot.join("lib/rustlib").join(target).exists())
}

/// What runs where Orla is WebAssembly, below the session.
const PACKAGES: &[&str] = &[
    "orla-abi",
    "orla-host",
    "orla-ir",
    "orla-rules",
    "orla-kernel",
    "orla-graph",
    "orla-language",
    "orla-compiler",
    "orla-scene",
    "orla-cpu",
    "orla-conformance",
    "orla-jit",
];

#[test]
fn thirty_two_bits_and_the_jit_s_webassembly_compute_what_the_interpreter_does() {
    let Some(node) = node() else {
        eprintln!("skipped: no node on this machine");
        return;
    };
    if !has_target("wasm32-wasip1") {
        eprintln!("skipped: no wasm32-wasip1 target (rustup target add wasm32-wasip1)");
        return;
    }
    let runner = format!(
        "{} {}",
        node.display(),
        root().join("bridges/web/wasi.mjs").display()
    );
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    // A target directory of its own, so the build does not wait on the
    // one running this test.
    let ran = Command::new(cargo)
        .current_dir(root())
        .env("CARGO_TARGET_WASM32_WASIP1_RUNNER", runner)
        .args(["test", "--release", "--features", "orla-jit/wasm-host"])
        .args(PACKAGES.iter().flat_map(|package| ["-p", package]))
        .args(["--target", "wasm32-wasip1", "--target-dir"])
        .arg(root().join("target/web"))
        .output()
        .expect("cargo runs");
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&ran.stdout),
        String::from_utf8_lossy(&ran.stderr)
    );
    assert!(
        ran.status.success(),
        "the JIT's tests on WebAssembly:\n{said}"
    );
    // Every test file ran: one run a file at least, and none skipped.
    let results = said.matches("test result: ok").count();
    assert!(
        results >= PACKAGES.len() + 3,
        "fewer test runs than there are:\n{said}"
    );
}
