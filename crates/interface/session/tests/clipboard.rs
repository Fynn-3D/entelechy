//! An input `= clipboard` holds the clipboard's text, a character a row
//! and a 0 after the last; an output `to clipboard` puts its characters
//! there, to its first 0, when it changes. The session's own clipboard
//! needs no allowing; the system's, given by the caller, does.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use entelechy_kernel::{Context, Value};
use entelechy_session::{Allowance, Clipboard, Session};

const SCENE: &str = "
kernel t.copy@1 { requires { field.read; field.write; } input text: Rows<Int>; output result: Rows<Int>; compute { return sample(text); } }
kernel t.word@1 { requires { field.write; } param which: Int = 0; output result: Rows<Int>;
    compute { return if which == 0 { [104, 105, 0, 0][element()] } else if which == 1 { [20320, 0, 0, 0][element()] } else { 0 }; } }
param which: Int = 0;
input pasted: Rows<Int> = clipboard;
node seen: t.copy@1 { text <- pasted; }
node word: t.word@1 { which = which; domain = 4; }
output result <- seen.result;
output copied <- word.result to clipboard;
";

fn scratch(name: &str) -> PathBuf {
    let dir =
        std::env::temp_dir().join(format!("entelechy-clipboard-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// What `result` holds after a sync and an evaluation, as text to its 0.
fn pasted(session: &mut Session) -> (String, Vec<u32>) {
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    let words = session.field(handle).unwrap().values().to_vec();
    let text = words
        .iter()
        .take_while(|&&w| w != 0)
        .filter_map(|&w| char::from_u32(w))
        .collect();
    (text, words)
}

/// A clipboard the test watches, standing for the system's.
#[derive(Clone, Default)]
struct Watched(Arc<Mutex<Option<String>>>);

impl Clipboard for Watched {
    fn text(&mut self) -> Option<String> {
        self.0.lock().unwrap().clone()
    }

    fn put(&mut self, text: &str) -> Result<(), String> {
        *self.0.lock().unwrap() = Some(text.to_owned());
        Ok(())
    }
}

#[test]
fn the_sessions_clipboard_is_read_and_written_as_text() {
    let dir = scratch("own");
    let mut session = open(&dir);
    // The output's word is put on the clipboard after the first frame;
    // the input reads it at the next sync.
    let (first, words) = pasted(&mut session);
    assert_eq!((first.as_str(), words.last()), ("", Some(&0)));
    assert_eq!(pasted(&mut session).0, "hi");
    // Text put on it from elsewhere, any character, is read at once.
    session.place_clipboard("héllo 你").unwrap();
    let (text, words) = pasted(&mut session);
    assert_eq!(text, "héllo 你");
    assert_eq!(words.len(), "héllo 你".chars().count() + 1);
    // A changed output is put again.
    session.set_param("which", Value::Int(1)).unwrap();
    pasted(&mut session);
    assert_eq!(pasted(&mut session).0, "你");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_systems_clipboard_is_read_and_written_only_as_allowed() {
    let dir = scratch("system");
    let mut session = open(&dir);
    let system = Watched::default();
    *system.0.lock().unwrap() = Some("outside".into());
    session.give_clipboard(Box::new(system.clone()));
    // Neither allowed: nothing read, nothing written, each said.
    let mut said: Vec<String> = session.sync().iter().map(|e| e.to_string()).collect();
    session.evaluate(&Context::default()).unwrap();
    said.extend(session.sync().iter().map(|e| e.to_string()));
    assert!(
        said.iter().any(|e| e.contains("clipboard.read")),
        "{said:?}"
    );
    assert!(
        said.iter().any(|e| e.contains("clipboard.write")),
        "{said:?}"
    );
    assert_eq!(pasted(&mut session).0, "");
    assert_eq!(system.0.lock().unwrap().as_deref(), Some("outside"));
    // Allowed, the system's text is read and the output's put.
    session.allow(&Allowance::ClipboardRead).unwrap();
    assert_eq!(pasted(&mut session).0, "outside");
    session.allow(&Allowance::ClipboardWrite).unwrap();
    session.set_param("which", Value::Int(1)).unwrap();
    pasted(&mut session);
    assert_eq!(system.0.lock().unwrap().as_deref(), Some("你"));
    // Text copied elsewhere stays while the output's text is the same;
    // the output's text again after none is put again.
    *system.0.lock().unwrap() = Some("elsewhere".into());
    session.set_param("which", Value::Int(1)).unwrap();
    pasted(&mut session);
    assert_eq!(pasted(&mut session).0, "elsewhere");
    session.set_param("which", Value::Int(2)).unwrap();
    pasted(&mut session);
    session.set_param("which", Value::Int(1)).unwrap();
    pasted(&mut session);
    assert_eq!(system.0.lock().unwrap().as_deref(), Some("你"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_clipboard_is_text_and_written_by_one_output() {
    let dir = scratch("refused");
    for (scene, said) in [
        (
            "input pasted: Rows<Float> = clipboard;\noutput result <- pasted;\n",
            "`Rows<Int>`",
        ),
        (
            "kernel t.w@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\nnode a: t.w@1 { domain = 1; }\nnode b: t.w@1 { domain = 1; }\noutput x <- a.result to clipboard;\noutput y <- b.result to clipboard;\n",
            "written by two outputs",
        ),
    ] {
        std::fs::write(dir.join("scene.esn"), scene).unwrap();
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
        let events: Vec<String> = session.sync().iter().map(|e| e.to_string()).collect();
        assert!(events.iter().any(|e| e.contains(said)), "{events:?}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_replay_puts_text_on_the_clipboard() {
    let dir = scratch("replay");
    let text = "scene scene.esn\nframe 0.0\nclipboard \"a#b\"\nframe 0.05\nexpect seen.result[0] = 97\nexpect seen.result[1] = 35\nexpect seen.result[3] = 0\n";
    let replay = entelechy_session::replay::parse(text, &dir).unwrap();
    let runtime = Session::runtime_for("cpu", entelechy_session::Limits::default()).unwrap();
    let outcome = entelechy_session::replay::run(&replay, runtime).unwrap();
    assert_eq!(
        (outcome.checks, outcome.failures.len()),
        (3, 0),
        "{outcome:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
