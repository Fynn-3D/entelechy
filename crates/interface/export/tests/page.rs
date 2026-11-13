//! A scene exported as one page: what is gathered and how it is named,
//! the packing the page's template reads, and the template filled.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use entelechy_export::page;

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

/// What a scene reads as it loads is gathered under `play/`, counted from
/// the directory all of it is in; the scene is among it; and no directory
/// of this machine is written into a name.
#[test]
fn a_scene_is_gathered_with_what_it_reads() {
    let scene = repository().join("examples/ant/scene.esn");
    let gathered = page::gather(&scene).expect("the ant example loads");
    assert_eq!(gathered.scene, "play/scene.esn");
    let names: Vec<&str> = gathered
        .files
        .iter()
        .map(|(name, _)| name.as_str())
        .collect();
    assert!(names.contains(&"play/scene.esn"), "{names:?}");
    assert!(
        names
            .iter()
            .any(|n| n.starts_with("play/kernels/") && n.ends_with(".ent")),
        "{names:?}"
    );
    for name in &names {
        assert!(
            name.starts_with("play/") || name.starts_with("packages/"),
            "{name}"
        );
        assert!(!name.contains("..") && !name.starts_with('/'), "{name}");
    }
    let (_, text) = gathered
        .files
        .iter()
        .find(|(name, _)| name == "play/scene.esn")
        .expect("the scene");
    assert_eq!(text, &std::fs::read(&scene).expect("the scene on disk"));
}

/// A scene that does not load is refused with why, not packed.
#[test]
fn a_scene_that_does_not_load_is_refused() {
    let missing = repository().join("examples/ant/no-such-scene.esn");
    assert!(page::gather(&missing).is_err());
}

/// The files are packed as the template says: a count, then for each a
/// path's length, the path, the bytes' length, the bytes.
#[test]
fn files_are_packed_one_after_another() {
    let files = vec![
        ("play/a.esn".to_string(), b"abc".to_vec()),
        ("play/b".to_string(), Vec::new()),
    ];
    let packed = page::archive(&files);
    let word =
        |at: usize| u32::from_le_bytes(packed[at..at + 4].try_into().expect("four bytes")) as usize;
    assert_eq!(word(0), 2);
    assert_eq!(word(4), 10);
    assert_eq!(&packed[8..18], b"play/a.esn");
    assert_eq!(word(18), 3);
    assert_eq!(&packed[22..25], b"abc");
    assert_eq!(word(25), 6);
    assert_eq!(&packed[29..35], b"play/b");
    assert_eq!(word(35), 0);
    assert_eq!(packed.len(), 39);
}

/// An include is the file beside the template (a pack the same, packed), a
/// slot is the exporter's;
/// text put inside a script cannot end the script.
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
/// was looked for and where.
#[test]
fn a_page_needs_entelechy_for_the_web() {
    let scene = repository().join("examples/ant/scene.esn");
    let nowhere = std::env::temp_dir().join("entelechy-no-web-here");
    let why = page::page(&scene, &nowhere, None, None)
        .err()
        .expect("refused");
    assert!(
        why.contains("single.html") && why.contains("--web"),
        "{why}"
    );
}

/// A font the page carries keeps the characters named and no others; a
/// file that is not a font is left as it is.
#[test]
fn a_font_is_carried_with_the_characters_named() {
    use entelechy_import::font::{read, sample};
    let whole = sample::latin();
    let (all, _) = read(&whole, None).expect("the sample font");
    let codes = |fields: &entelechy_import::font::Fields| -> Vec<u32> {
        fields
            .glyphs
            .values()
            .chunks_exact(8)
            .map(|row| row[0])
            // (The first row, of code -1, is the glyph for a character the font lacks.)
            .filter(|&code| code as i32 >= 0)
            .collect()
    };
    let mapped = codes(&all);
    assert!(mapped.len() > 1, "the sample maps more than one character");
    let one = char::from_u32(mapped[0]).expect("a character").to_string();
    let kept = page::font_of("play/fonts/sample.ttf", &whole, &one)
        .expect("read and written")
        .expect("a font");
    let (few, _) = read(&kept, None).expect("what was carried is a font");
    assert_eq!(codes(&few), vec![mapped[0]]);
    assert!(
        page::font_of("play/words.txt", b"A", "A")
            .expect("not a font")
            .is_none()
    );
}
