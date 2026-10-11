//! Entelechy as WebAssembly (32-bit `usize`, a JIT emitting WebAssembly): the
//! tests of the core, the front end, the interpreter and the JIT, built
//! for `wasm32-wasip1` in release and run in node, which hosts the JIT's
//! code as a page does (`bridges/web/wasi.mjs`). Release keeps frames
//! within the engine's stack at the deepest nesting. Skipped with a note
//! without node or the target.

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

/// What runs where Entelechy is WebAssembly, below the session.
const PACKAGES: &[&str] = &[
    "entelechy-abi",
    "entelechy-host",
    "entelechy-ir",
    "entelechy-rules",
    "entelechy-kernel",
    "entelechy-graph",
    "entelechy-language",
    "entelechy-compiler",
    "entelechy-scene",
    "entelechy-cpu",
    "entelechy-conformance",
    "entelechy-jit",
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
    // Its own target directory, independent of the running build.
    let ran = Command::new(cargo)
        .current_dir(root())
        .env("CARGO_TARGET_WASM32_WASIP1_RUNNER", runner)
        .args(["test", "--release", "--features", "entelechy-jit/wasm-host"])
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
    // Every test file ran, none skipped.
    let results = said.matches("test result: ok").count();
    assert!(
        results >= PACKAGES.len() + 3,
        "fewer test runs than there are:\n{said}"
    );
}
