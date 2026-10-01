//! A served session reads only where its first scene may: every read it
//! makes, however deep the document that names the file, is judged where
//! the path lands after its links and `..`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use orla_session::{Event, Session};

const SCENE: &str = "node start: Rows<Int> { domain = 1; value = 1; }
output result <- start.result;
";

const FAR: &str = "kernel far.away@1 { requires { field.write; } output result: Rows<Int>; compute { return 7; } }";

/// A 1x1 picture, as a P6 file.
const PICTURE: &[u8] = b"P6\n1 1\n255\n\xff\x80\x00";

/// A directory holding `app/scene.osn` and, outside it, `outside/` with a
/// Kernel, a picture and a directory.
fn built(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-reads-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("app")).unwrap();
    std::fs::create_dir_all(dir.join("outside/data")).unwrap();
    std::fs::write(dir.join("app/scene.osn"), SCENE).unwrap();
    std::fs::write(dir.join("outside/far.orla"), FAR).unwrap();
    std::fs::write(dir.join("outside/secret.ppm"), PICTURE).unwrap();
    dir
}

/// A session of `app/scene.osn` with its reads held, first loaded.
fn served(dir: &Path) -> Session {
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let store = Arc::new(orla_session::Disk::default());
    let mut session = Session::open_in(store, &dir.join("app/scene.osn"), None, runtime);
    session.restrict_reads().unwrap();
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// Composes the scene with `extra` added and syncs; the events.
fn composed(session: &mut Session, extra: &str) -> Vec<Event> {
    let text = format!("{SCENE}{extra}");
    match session.write_scene(&text).unwrap() {
        Ok(()) => session.sync(),
        Err(refused) => vec![Event::SceneRejected(refused)],
    }
}

/// Whether input `name` was read.
fn loaded(events: &[Event], name: &str) -> bool {
    events
        .iter()
        .any(|event| matches!(event, Event::Loaded { name: loaded, .. } if loaded == name))
}

#[test]
fn a_placed_graphs_own_use_is_held() {
    let dir = built("nested");
    let mut session = served(&dir);
    std::fs::write(
        dir.join("app/sub.osn"),
        "use \"../outside/far.orla\";\nnode f: far.away@1 { }\noutput result <- f.result;\n",
    )
    .unwrap();
    let events = composed(&mut session, "node s: graph \"sub.osn\" { }\n");
    let published = session
        .runtime()
        .kernels()
        .any(|version| version.id().to_string().starts_with("far.away"));
    assert!(!published, "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_package_made_outside_is_not_read() {
    let dir = built("package");
    let mut session = served(&dir);
    let package = dir.join("outside/made.orlapkg");
    std::fs::create_dir_all(&package).unwrap();
    std::fs::write(package.join("package.osn"), "").unwrap();
    std::fs::write(package.join("secret.ppm"), PICTURE).unwrap();
    let events = composed(
        &mut session,
        "input pic: Image = \"../outside/made.orlapkg/secret.ppm\";\n",
    );
    assert!(!loaded(&events, "pic"), "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A package used by path from outside is refused, saying how to use
/// it: by name, its directory in `ORLA_PACKAGES`.
#[test]
fn a_package_outside_is_refused_naming_the_way_in() {
    let dir = built("package-use");
    let mut session = served(&dir);
    let package = dir.join("outside/made.orlapkg");
    std::fs::create_dir_all(package.join("kernels")).unwrap();
    std::fs::write(
        package.join("package.osn"),
        "package made@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
    )
    .unwrap();
    std::fs::write(package.join("kernels/far.orla"), FAR).unwrap();
    let events = composed(&mut session, "use \"../outside/made.orlapkg\";\n");
    let said = events.iter().any(|event| {
        matches!(event, Event::SceneRejected(why)
            if why.message.contains("`use \"made.orlapkg\";`") && why.message.contains("ORLA_PACKAGES"))
    });
    assert!(said, "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_link_and_dot_dot_are_judged_where_they_land() {
    let dir = built("link");
    let mut session = served(&dir);
    std::os::unix::fs::symlink(dir.join("outside/data"), dir.join("app/link")).unwrap();
    // Folded as text, `link/../secret.ppm` is `app/secret.ppm`; followed,
    // it is `outside/secret.ppm`.
    let events = composed(&mut session, "input pic: Image = \"link/../secret.ppm\";\n");
    assert!(!loaded(&events, "pic"), "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_linked_frame_is_held() {
    let dir = built("frame");
    let mut session = served(&dir);
    std::fs::create_dir_all(dir.join("app/clip")).unwrap();
    std::os::unix::fs::symlink(dir.join("outside/secret.ppm"), dir.join("app/clip/0.ppm")).unwrap();
    let mut events = composed(
        &mut session,
        "input clip: Image = frames \"clip/\" at 1;\nnode shown: kernel {
    requires { field.read; field.write; }
    input p: Image <- clip;
    output result: Image;
    compute { return sample(p); }
}
output view <- shown.result;
",
    );
    let evaluated = session.evaluate(&Default::default());
    events.extend(session.sync());
    let refused = events.iter().any(|event| {
        matches!(event, Event::Rejected { error, .. } if error.to_string().contains("outside"))
    });
    assert!(evaluated.is_err() && refused, "{evaluated:?} {events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[cfg(unix)]
#[test]
fn a_write_through_a_link_and_dot_dot_is_judged_where_it_lands() {
    let dir = built("write");
    let mut session = served(&dir);
    std::fs::create_dir_all(dir.join("app/out")).unwrap();
    session.allow_writes(&dir.join("app/out")).unwrap();
    std::os::unix::fs::symlink(dir.join("outside/data"), dir.join("app/out/link")).unwrap();
    let written = session.write_file(&dir.join("app/out/link/../x.txt"), b"x");
    assert!(written.is_err(), "{written:?}");
    assert!(!dir.join("outside/x.txt").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_first_scenes_own_inputs_stay_readable() {
    let dir = built("own");
    std::fs::write(
        dir.join("app/scene.osn"),
        format!("{SCENE}input pic: Image = \"../outside/secret.ppm\";\n"),
    )
    .unwrap();
    let runtime = Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let store = Arc::new(orla_session::Disk::default());
    let mut session = Session::open_in(store, &dir.join("app/scene.osn"), None, runtime);
    session.restrict_reads().unwrap();
    let events = session.sync();
    assert!(loaded(&events, "pic"), "{events:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
