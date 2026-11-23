//! A scene exported as one page: how it compiles into a binary bundle with
//! no host paths, how font nodes are subsetted, how the template is filled,
//! and how invalid inputs are refused.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_export::page;
use entelechy_session::bundle::Bundle;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// A scene compiles into a binary bundle that decodes back cleanly and
/// holds no directory of this machine.
#[test]
fn a_scene_compiles_into_a_bundle_without_host_paths() {
    let root = repository().canonicalize().unwrap_or_else(|_| repository());
    let scene = root.join("examples/ant/scene.esn");
    let compiled = page::compile(&scene, None).expect("the ant example compiles");
    assert!(compiled.modules > 0);
    let root_bytes = root.to_string_lossy().into_owned();
    assert!(
        !compiled
            .bundle
            .windows(root_bytes.len())
            .any(|w| w == root_bytes.as_bytes()),
        "no host path in the bundle"
    );
    let decoded = Bundle::decode(&compiled.bundle).expect("bundle decodes");
    assert_eq!(decoded.modules.len(), compiled.modules);
    assert_eq!(decoded.inputs.len(), compiled.inputs);
}

/// A scene that does not load is refused with why, not packed.
#[test]
fn a_scene_that_does_not_load_is_refused() {
    let missing = repository().join("examples/ant/no-such-scene.esn");
    assert!(page::compile(&missing, None).is_err());
}

/// An include is the file beside the template (a pack the same, packed), a
/// slot is the exporter's; text put inside a script cannot end the script.
#[test]
fn a_template_is_filled() {
    let dir = std::env::temp_dir().join(format!("entelechy-page-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("a directory");
    std::fs::write(dir.join("part.js"), "let end = \"</script>\";").expect("a file");
    let mut slots = BTreeMap::new();
    slots.insert("title", "ant".to_string());
    let filled = page::fill(
        "<title>@@slot:title@@</title><script>@@include:part.js@@</script>",
        &dir,
        &slots,
    )
    .expect("filled");
    assert_eq!(
        filled,
        "<title>ant</title><script>let end = \"<\\/script>\";</script>"
    );
    // (A file packed is a zlib stream in UTF-8 base122: nothing of the file shows.)
    let packed = page::fill("@@pack:part.js@@", &dir, &slots).expect("packed");
    assert!(!packed.contains('<') && !packed.contains("end"), "{packed}");
    let unzipped = entelechy_formats::inflate::unzlib(
        &entelechy_formats::base122::unbase122(&packed).expect("base122"),
        usize::MAX,
    )
    .expect("zlib");
    assert_eq!(unzipped, b"let end = \"</script>\";");
    assert!(
        page::fill("@@slot:other@@", &dir, &slots).is_err(),
        "a slot nobody fills"
    );
    assert!(
        page::fill("@@include:none.js@@", &dir, &slots).is_err(),
        "a file not there"
    );
    assert!(
        page::fill("@@what@@", &dir, &slots).is_err(),
        "neither an include nor a slot"
    );
    assert!(
        page::fill("@@slot:title", &dir, &slots).is_err(),
        "a mark not closed"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// Without Entelechy for the web where `--web` says, the error names what
/// was looked for and where; `--backend cpu` is refused for a page.
#[test]
fn a_page_needs_entelechy_for_the_web_and_a_page_backend() {
    let scene = repository().join("examples/ant/scene.esn");
    let nowhere = std::env::temp_dir().join("entelechy-no-web-here");
    let why = page::page(&scene, &nowhere, None, None)
        .err()
        .expect("refused");
    assert!(
        why.contains("single.html") && why.contains("--web"),
        "{why}"
    );
    let cpu = page::page_with_pointer(&scene, &nowhere, Some("cpu"), None, page::PointerMode::Lock)
        .err()
        .expect("cpu refused");
    assert!(cpu.contains("gpu or jit"), "{cpu}");
    assert_eq!(page::PointerMode::Free.as_str(), "free");
    assert_eq!(page::PointerMode::Hold.as_str(), "hold");
    assert_eq!(page::PointerMode::Lock.as_str(), "lock");
    assert_eq!(
        page::default_title(Path::new("examples/voxel/scene.esn")),
        "voxel"
    );
    assert_eq!(page::default_title(Path::new("fps/graph/world.esn")), "fps");
    assert_eq!(page::default_title(Path::new("demos/harbor.esn")), "harbor");
}

/// A scene's `font` node baked with `--chars` keeps only the characters
/// named (plus the missing-character row at -1).
#[test]
fn a_font_node_is_baked_with_the_characters_named() {
    use entelechy_import::font::{read, sample};
    let dir = std::env::temp_dir().join(format!("entelechy-page-font-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let whole = sample::latin();
    let (all, _) = read(&whole, None).unwrap();
    let mapped: Vec<u32> = all
        .glyphs
        .values()
        .chunks_exact(8)
        .map(|row| row[0])
        .filter(|&code| code as i32 >= 0)
        .collect();
    assert!(mapped.len() > 1);
    let one = char::from_u32(mapped[0]).unwrap().to_string();

    std::fs::write(dir.join("sample.ttf"), &whole).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "struct Letterform { code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2; }\n\
         kernel f.codes@1 { requires { field.write; } output result: Rows<Int>; compute { return 73; } }\n\
         kernel f.show@1 { requires { field.read; field.write; } input glyphs: Rows<Letterform>; output result: Rows<Int>; compute { return at(glyphs, 0).code; } }\n\
         node codes: f.codes@1 { domain = 1; }\n\
         node face: font \"sample.ttf\" { codes <- codes.result; }\n\
         node show: f.show@1 { glyphs <- face.glyphs; domain = 1; }\n\
         output result <- show.result;\n",
    )
    .unwrap();

    let compiled = page::compile(&dir.join("scene.esn"), Some(&one)).expect("compiled");
    assert_eq!(compiled.fonts.len(), 1);
    let decoded = Bundle::decode(&compiled.bundle).expect("bundle decodes");
    assert_eq!(decoded.fonts.len(), 1);
    assert_eq!(decoded.fonts[0].mapped_codes(), vec![mapped[0]]);

    // Without `--chars`, a static `codes` producer subsets the font on its own.
    let auto = page::compile(&dir.join("scene.esn"), None).expect("auto-subsetted");
    let auto_decoded = Bundle::decode(&auto.bundle).expect("bundle decodes");
    assert_eq!(auto_decoded.fonts[0].mapped_codes(), vec![73]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// Runtime save files (`output ... to "save.field"` paired with an `input ... else`
/// fallback) on the exporter's disk are not baked into an exported page.
#[test]
fn runtime_save_files_are_not_baked_into_exported_pages() {
    use entelechy_host::{Domain, FieldData, Scalar, Schema};
    let dir = std::env::temp_dir().join(format!("entelechy-page-save-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let stale_save = FieldData::from_storage(
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(1),
        vec![999u32],
    )
    .unwrap();
    std::fs::write(
        dir.join("save.field"),
        entelechy_formats::file::encode_data(&stale_save),
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel s.fresh@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }\n\
         kernel s.id@1 { requires { field.read; field.write; } input v: Rows<Int>; output result: Rows<Int>; compute { return at(v, 0); } }\n\
         node fresh: s.fresh@1 { domain = 1; }\n\
         input saved: Rows<Int> = \"save.field\" else fresh.result;\n\
         node shown: s.id@1 { v <- saved; domain = 1; }\n\
         output result <- shown.result;\n\
         output kept <- shown.result to \"save.field\";\n",
    )
    .unwrap();
    let compiled = page::compile(&dir.join("scene.esn"), None).expect("compiled");
    assert_eq!(compiled.inputs, 0, "stale save.field must not be bundled");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_exporter_writes_a_standalone_bundle_from_a_scene() {
    let dir = std::env::temp_dir().join(format!("entelechy-bundle-cli-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    let out = dir.join("ant.bundle");
    let scene = repository().join("examples/ant/scene.esn");
    let ran = std::process::Command::new(env!("CARGO_BIN_EXE_entelechy-export"))
        .arg(&scene)
        .arg("--out")
        .arg(&out)
        .output()
        .expect("entelechy-export runs");
    assert!(
        ran.status.success(),
        "entelechy-export failed: {}",
        String::from_utf8_lossy(&ran.stderr)
    );
    let bytes = std::fs::read(&out).expect("bundle file written");
    let decoded = Bundle::decode(&bytes).expect("bundle decodes");
    assert!(!decoded.modules.is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}
