//! Declarations shared through `use`: one declaration reached from two
//! files is one, wherever written; two different ones of a name are
//! refused.

use std::path::PathBuf;

use orla_session::declarations_for;

fn scratch(name: &str, files: &[(&str, &str)]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-sources-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    for (file, text) in files {
        std::fs::write(dir.join(file), text).unwrap();
    }
    dir
}

fn uses(text: &str) -> Vec<orla_language::Use> {
    orla_language::parse_document(text).unwrap().uses
}

#[test]
fn one_declaration_written_in_two_files_is_one() {
    let dir = scratch(
        "same",
        &[
            ("a.orla", "struct Surface { depth: Float; id: Int; }\n"),
            // The same fields, written further down its file.
            (
                "b.orla",
                "// another library\n\n\nstruct Surface { depth: Float; id: Int; }\n",
            ),
        ],
    );
    let (structs, _, _) = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";\nuse \"b.orla\";"),
    )
    .unwrap();
    assert_eq!(structs.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn two_declarations_of_one_name_are_refused() {
    let dir = scratch(
        "different",
        &[
            ("a.orla", "struct Surface { depth: Float; id: Int; }\n"),
            ("b.orla", "struct Surface { depth: Float; }\n"),
        ],
    );
    let error = declarations_for(
        &orla_session::Disk::default(),
        &dir,
        &uses("use \"a.orla\";\nuse \"b.orla\";"),
    )
    .unwrap_err();
    assert!(error.contains("declared differently"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A file added to a directory a scene uses is published by the next
/// sync, and one taken away is retired.
#[test]
fn a_used_directory_gains_and_loses_files_between_syncs() {
    let dir = scratch(
        "gains",
        &[(
            "scene.osn",
            "use \"kernels/\";\nnode x: lib.a@1 { domain = 4; }\noutput result <- x.result;\n",
        )],
    );
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    let kernel = |name: &str| {
        format!(
            "kernel lib.{name}@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 1.0; }} }}\n"
        )
    };
    std::fs::write(dir.join("kernels/a.orla"), kernel("a")).unwrap();
    let runtime =
        orla_session::Session::runtime_for("cpu", orla_session::Limits::default()).unwrap();
    let mut session = orla_session::Session::open_in(
        std::sync::Arc::new(orla_session::Disk::default()),
        &dir.join("scene.osn"),
        None,
        runtime,
    );
    let published = |session: &orla_session::Session, name: &str| {
        session
            .runtime()
            .kernel(&format!("lib.{name}@1").parse().unwrap())
            .is_some()
    };
    let events = session.sync();
    assert!(published(&session, "a"), "{events:?}");
    assert!(!published(&session, "b"));
    // Idle syncs in between keep what they reached.
    session.sync();
    assert!(published(&session, "a"));
    std::fs::write(dir.join("kernels/b.orla"), kernel("b")).unwrap();
    let events = session.sync();
    assert!(published(&session, "b"), "{events:?}");
    std::fs::remove_file(dir.join("kernels/b.orla")).unwrap();
    let events = session.sync();
    assert!(!published(&session, "b"), "{events:?}");
    assert!(published(&session, "a"));
    let _ = std::fs::remove_dir_all(&dir);
}
