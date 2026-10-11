//! `--help` and `-h` print the Player's usage and succeed, opening no
//! window.

use std::process::{Command, Stdio};

#[test]
fn help_prints_the_usage_and_succeeds() {
    for flag in ["--help", "-h"] {
        let output = Command::new(env!("CARGO_BIN_EXE_entelechy-player"))
            .arg(flag)
            .stdin(Stdio::null())
            .output()
            .expect("the Player runs");
        assert!(output.status.success(), "{flag}");
        let text = String::from_utf8_lossy(&output.stdout);
        assert!(
            text.starts_with("usage:") && text.contains("entelechy-player <scene.esn>"),
            "{flag}: {text}"
        );
    }
}
