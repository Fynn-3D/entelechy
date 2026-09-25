//! What documents share through `use`: the same declaration reached from
//! two files is one, wherever each file wrote it, and two different
//! declarations of one name are refused.

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
        &orla_session::Disk,
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
        &orla_session::Disk,
        &dir,
        &uses("use \"a.orla\";\nuse \"b.orla\";"),
    )
    .unwrap_err();
    assert!(error.contains("declared differently"), "{error}");
    let _ = std::fs::remove_dir_all(&dir);
}
