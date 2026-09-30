//! A rig is five kinds of row, written by `orla import` from glTF, by the
//! Blender bridge from an armature, or by anyone. The rigs here are held
//! to [the contract](../../../docs/RIGS.md): the layouts, and that the
//! `skin` package reads what the importer writes.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{rows_keep_the_contract, schema_of};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// Fresh `orla import` output keeps the contract's layouts, and a scene
/// binding it to the `skin` Kernels evaluates.
#[test]
fn the_importer_writes_what_the_package_reads() {
    let importer = Path::new(env!("CARGO_BIN_EXE_orla"))
        .parent()
        .unwrap()
        .join("orla-import");
    if !importer.exists() {
        eprintln!("skipped: orla-import is not built beside orla");
        return;
    }
    let dir = std::env::temp_dir().join(format!("orla-rig-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let import = Command::new(&importer)
        .arg(root().join("examples/skin/source/arm.glb"))
        .arg("--out")
        .arg(&dir)
        .args(["--rate", "24"])
        .output()
        .unwrap();
    assert!(
        import.status.success(),
        "{}",
        String::from_utf8_lossy(&import.stderr)
    );
    let found = rows_keep_the_contract(&dir);
    assert_eq!(found.len(), 6, "five kinds, two of them clips: {found:?}");

    // The package's Kernels take these layouts; the binding check refuses
    // a mismatch on either side, naming both.
    let root = root().canonicalize().unwrap();
    let scene = dir.join("pose.osn");
    std::fs::write(
        &scene,
        format!(
            "use \"{root}/packages/skin.orlapkg\";\n\
             input mesh: Rows<Point> = \"arm.vertices.field\";\n\
             input weights: Rows<Bound> = \"arm.weights.field\";\n\
             input skeleton: Skeleton = \"arm.skeleton.field\";\n\
             input swing: Clip = \"arm.swing.clip.field\";\n\
             node clock: kernel {{ requires {{ field.write; }} output result: Rows<Float>; \
             domain = 1; compute {{ return 0.25; }} }}\n\
             node posed: graph \"{root}/packages/skin.orlapkg/graphs/pose.osn\" {{\n\
             mesh <- mesh; weights <- weights; skeleton <- skeleton; clip <- swing;\n\
             at <- clock.result; rate = 24.0; }}\n\
             output result <- posed.result;\n",
            root = root.display(),
        ),
    )
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("run")
        .arg(&scene)
        .args(["--backend", "cpu"])
        .arg("--save")
        .arg(format!(
            "posed.posed.result={}",
            dir.join("posed.field").display()
        ))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "the package could not read what the importer wrote:\n{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    // The output mesh has the input's shape.
    assert_eq!(
        schema_of(&dir.join("posed.field")),
        "f,f,f,f,f,f,f,f",
        "a posed mesh is a mesh"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
