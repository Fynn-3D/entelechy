//! The bridges are clients outside the workspace, each with a runner of
//! its own: a page driven headless through Chrome's DevTools protocol,
//! and a Godot project run with `-s`. Both are run from here when the
//! tools they need are on this machine, and skipped with a word when
//! they are not, so a change to the protocol that breaks a bridge
//! breaks this rather than waiting for someone to open one.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The workspace root, from this crate.
fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A tool this machine has, by the paths it is usually at, or nothing.
fn tool(paths: &[&str]) -> Option<PathBuf> {
    paths
        .iter()
        .map(PathBuf::from)
        .find(|path| path.exists())
        .or_else(|| {
            let found = Command::new("which").arg(paths[0]).output().ok()?;
            found
                .status
                .success()
                .then(|| PathBuf::from(String::from_utf8_lossy(&found.stdout).trim()))
        })
}

/// Runs a bridge's own runner and says what it said. The runners print
/// a line for every thing they checked and end with `RESULT: ok`.
fn passed(mut command: Command, what: &str, marker: &str) {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("{what}: {error}"));
    let said = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.status.success() && said.contains("RESULT: ok"),
        "{what} did not pass:\n{said}"
    );
    // One of the things it checked, so a runner that said `ok` without
    // running is not taken for a pass.
    assert!(
        said.contains(marker),
        "{what} did not check `{marker}`:\n{said}"
    );
}

#[test]
fn the_web_bridge_passes_its_own_test() {
    let Some(node) = tool(&["node", "/usr/local/bin/node", "/opt/homebrew/bin/node"]) else {
        eprintln!("skipped: no node on this machine");
        return;
    };
    let Some(chrome) = tool(&[
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "google-chrome",
        "chromium",
    ]) else {
        eprintln!("skipped: no Chrome on this machine");
        return;
    };
    let mut command = Command::new(node);
    command
        .current_dir(root())
        .arg("bridges/web/test.mjs")
        .arg("--orla")
        .arg(env!("CARGO_BIN_EXE_orla"))
        .arg("--chrome")
        .arg(chrome);
    passed(command, "the web bridge", "voxel: a click places a voxel");
}

#[test]
fn the_godot_bridge_passes_its_own_test() {
    let Some(godot) = tool(&["godot", "/usr/local/bin/godot"]) else {
        eprintln!("skipped: no Godot on this machine");
        return;
    };
    let mut command = Command::new(godot);
    command
        .current_dir(root())
        .args(["--headless", "--path", "bridges/godot", "-s", "test.gd"])
        .args(["--", "--orla", env!("CARGO_BIN_EXE_orla")]);
    passed(
        command,
        "the Godot bridge",
        "mesh: the torus is an ArrayMesh",
    );
}
