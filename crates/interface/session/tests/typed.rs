//! `typed(k)` reads the characters typed since the last evaluation: a
//! replay's `type` line, as many a frame as one holds, the rest on the
//! frames after; each frame's are its own.

use std::path::{Path, PathBuf};

use entelechy_session::{Session, replay};

const SCENE: &str = "kernel t.typed@1 {
    requires { field.write; input.read; }
    output result: Rows<Int>;
    compute { return typed(element() - 1); }
}
node chars: t.typed@1 { domain = 34; }
output result <- chars.result;
";

fn scene(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-typed-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn ran(dir: &Path, backend: &str, text: &str) -> Option<replay::Outcome> {
    let replay = replay::parse(text, dir).unwrap();
    let Ok(runtime) = Session::runtime_for(backend, entelechy_session::Limits::default()) else {
        eprintln!("no {backend} here; skipped");
        return None;
    };
    Some(replay::run(&replay, runtime).unwrap())
}

#[test]
fn typed_reads_each_frames_characters_on_every_backend() {
    let dir = scene("frames");
    // `typed(-1)` (element 0) and `typed(32)` (element 33) are outside.
    let text = "scene scene.esn
type \"你a#\\\"\" # a comment
frame 0
expect chars.result[0] = 0
expect chars.result[1] = 20320
expect chars.result[2] = 97
expect chars.result[3] = 35
expect chars.result[4] = 34
expect chars.result[5] = 0
expect chars.result[33] = 0
frame 0.1
expect chars.result[1] = 0
type \"abcdefghijklmnopqrstuvwxyzABCDEFGHIJ\"
frame 0.2
expect chars.result[1] = 97
expect chars.result[32] = 70
frame 0.3
expect chars.result[1] = 71
expect chars.result[4] = 74
expect chars.result[5] = 0
";
    for backend in ["cpu", "jit", "gpu"] {
        let Some(outcome) = ran(&dir, backend, text) else {
            continue;
        };
        assert_eq!(outcome.checks, 13, "{backend}");
        assert!(
            outcome.failures.is_empty(),
            "{backend}: {:?}",
            outcome.failures
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_type_line_takes_one_quoted_text() {
    let dir = Path::new(".");
    for wrong in ["type abc", "type \"open", "type \"a\" b", "type \"\\q\""] {
        let error = replay::parse(&format!("scene s.esn\n{wrong}\n"), dir).unwrap_err();
        assert_eq!(error.line, 2, "{wrong}");
        assert!(
            error.message.contains("`type`"),
            "{wrong}: {}",
            error.message
        );
    }
}
