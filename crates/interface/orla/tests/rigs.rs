//! A rig is five kinds of row, written by `orla import` from glTF, by the
//! Blender bridge from an armature, or by anyone. The rigs here are held
//! to [the contract](../../../docs/RIGS.md): the layouts, and that a
//! scene posing the rows reads what the importer writes.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use common::{arm, rows_keep_the_contract, schema_of};

/// A tool built beside `orla`, if it is.
fn beside(name: &str) -> Option<PathBuf> {
    let tool = Path::new(env!("CARGO_BIN_EXE_orla"))
        .parent()
        .unwrap()
        .join(name);
    tool.exists().then_some(tool)
}

/// A rig written as glTF and imported again keeps the contract's
/// layouts, and a scene posing it by those layouts evaluates.
#[test]
fn the_importer_writes_what_a_posing_scene_reads() {
    let (Some(exporter), Some(importer)) = (beside("orla-export"), beside("orla-import")) else {
        eprintln!("skipped: orla-export and orla-import are not built beside orla");
        return;
    };
    let dir = std::env::temp_dir().join(format!("orla-rig-contract-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let written = dir.join("written");
    std::fs::create_dir_all(&written).unwrap();
    arm(&written);
    let glb = dir.join("arm.glb");
    let rows: Vec<PathBuf> = [
        "vertices",
        "triangles",
        "weights",
        "skeleton",
        "swing.clip",
        "curl.clip",
    ]
    .iter()
    .map(|name| written.join(format!("arm.{name}.field")))
    .collect();
    let export = Command::new(&exporter)
        .args(&rows)
        .arg("--out")
        .arg(&glb)
        .args(["--rate", "24"])
        .output()
        .unwrap();
    assert!(
        export.status.success(),
        "{}",
        String::from_utf8_lossy(&export.stderr)
    );
    let import = Command::new(&importer)
        .arg(&glb)
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

    // Kernels posing by these layouts take them; the binding check
    // refuses a mismatch on either side, naming both.
    let kernels = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pose/pose.orla");
    let scene = dir.join("pose.osn");
    std::fs::write(
        &scene,
        format!(
            "use \"{kernels}\";\n\
             input mesh: Rows<Point> = \"arm.vertices.field\";\n\
             input weights: Rows<Bound> = \"arm.weights.field\";\n\
             input skeleton: Skeleton = \"arm.skeleton.field\";\n\
             input swing: Clip = \"arm.swing.clip.field\";\n\
             node clock: kernel {{ requires {{ field.write; }} output result: Rows<Float>; \
             domain = 1; compute {{ return 0.25; }} }}\n\
             node poses: pose.sample@1 {{ skeleton <- skeleton; clip <- swing; \
             at <- clock.result; rate = 24.0; domain = count(skeleton); }}\n\
             node palette: pose.palette@1 {{ poses <- poses.result; skeleton <- skeleton; \
             domain = count(skeleton); }}\n\
             node posed: pose.apply@1 {{ mesh <- mesh; weights <- weights; \
             palette <- palette.result; }}\n\
             output result <- posed.result;\n",
            kernels = kernels.display(),
        ),
    )
    .unwrap();
    let run = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("run")
        .arg(&scene)
        .args(["--backend", "cpu"])
        .arg("--save")
        .arg(format!(
            "posed.result={}",
            dir.join("posed.field").display()
        ))
        .output()
        .unwrap();
    assert!(
        run.status.success(),
        "a posing scene could not read what the importer wrote:\n{}{}",
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
