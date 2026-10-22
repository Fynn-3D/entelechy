//! A graph the scene places brings its own `use`s, whose Kernels a session
//! publishes with the scene's: a check compiles them too, so a fault in
//! one is said before the scene is run, as loading it says it.

use entelechy_kernel::{Context, Runtime};
use entelechy_session::Session;

/// The problems a check of the scene finds, and whether a session that
/// loads it refuses a document.
fn checked(name: &str, body: &str) -> (Vec<String>, bool) {
    let dir = std::env::temp_dir().join(format!(
        "entelechy-placed-uses-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("part/kernels")).unwrap();
    std::fs::write(
        dir.join("part/kernels/lib.ent"),
        format!("kernel p.fill@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ return {body}; }} }}\n"),
    )
    .unwrap();
    std::fs::write(
        dir.join("part/part.esn"),
        "use \"kernels/\";\nnode a: p.fill@1 { domain = 4; }\noutput result <- a.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "node part: graph \"part/part.esn\" {}\noutput result <- part.result;\n",
    )
    .unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let problems = checked
        .lines
        .iter()
        .filter(|(problem, _)| *problem)
        .map(|(_, line)| line.clone())
        .collect();
    let runtime = Runtime::with_backends(1 << 20, Box::new(entelechy_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    let refused = session
        .sync()
        .iter()
        .any(|event| format!("{event:?}").contains("Rejected"));
    let ran = session.evaluate(&Context::default()).is_ok();
    let _ = std::fs::remove_dir_all(&dir);
    (problems, refused || !ran)
}

#[test]
fn a_fault_in_a_document_a_placed_graph_uses_is_said_by_the_check() {
    let (problems, refused) = checked("fault", "domain().x");
    assert!(refused, "loading the scene refuses the document");
    assert!(
        problems
            .iter()
            .any(|line| line.contains("lib.ent") && line.contains("`.x`")),
        "the check says the fault where loading says it: {problems:?}"
    );
}

#[test]
fn a_sound_document_a_placed_graph_uses_passes_the_check() {
    let (problems, refused) = checked("sound", "element()");
    assert!(!refused, "the scene loads and runs");
    assert!(problems.is_empty(), "nothing to say: {problems:?}");
}

/// The problems a check finds in a scene that reads a tile map from bytes
/// kept as `u8` and hands it to a placed graph whose Kernel takes it so.
#[test]
fn a_narrow_bytes_input_handed_to_a_placed_graph_passes_the_check() {
    let dir = std::env::temp_dir().join(format!(
        "entelechy-placed-uses-narrow-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("part/kernels")).unwrap();
    std::fs::write(
        dir.join("part/kernels/lib.ent"),
        "kernel p.tile@1 { requires { field.read; field.write; } input tiles: Rows<Int as u8>; output result: Rows<Int>; compute { return at(tiles, element()); } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("part/part.esn"),
        "use \"kernels/\";\ninput tiles: Rows<Int as u8>;\nnode a: p.tile@1 { tiles <- tiles; domain = 4; }\noutput result <- a.result;\n",
    )
    .unwrap();
    std::fs::write(dir.join("tiles.bin"), [1u8, 2, 3, 4]).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "input tiles: Rows<Int as u8> = bytes \"tiles.bin\";\nnode part: graph \"part/part.esn\" { tiles <- tiles; }\noutput result <- part.result;\n",
    )
    .unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let problems: Vec<String> = checked
        .lines
        .iter()
        .filter(|(problem, _)| *problem)
        .map(|(_, line)| line.clone())
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    assert!(problems.is_empty(), "nothing to say: {problems:?}");
}
