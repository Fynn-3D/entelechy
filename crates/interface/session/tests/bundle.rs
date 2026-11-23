//! A scene packed into a binary [`Bundle`] decodes back to the same modules,
//! graph, inputs and font atlases, and plays word for word as the document
//! session it came from.

use std::path::{Path, PathBuf};

use entelechy_abi::{Domain, FieldData, Scalar, Schema};
use entelechy_kernel::{Context, Value};
use entelechy_session::bundle::Bundle;
use entelechy_session::{Limits, Session, Wait};

const SCENE: &str = "
struct Letterform { code: Int; advance: Int; first: Int; count: Int; lo: Vec2; hi: Vec2; }

kernel b.seed@1 {
    requires { field.write; }
    output result: Rows<Int>;
    compute { return 10 + element(); }
}

kernel b.step@1 {
    requires { field.read; field.write; }
    input prev: Rows<Int>;
    input weights: Rows<Int>;
    input title: Rows<Int>;
    input pasted: Rows<Int>;
    input glyphs: Rows<Letterform>;
    param gain: Int = 2;
    output result: Rows<Int>;
    compute {
        let i = element();
        let w = at(weights, i % count(weights));
        let t = at(title, i % count(title));
        let p = at(pasted, 0);
        let g = at(glyphs, i % count(glyphs)).advance;
        return at(prev, i) + w * gain + t + p + g;
    }
}

kernel b.codes@1 {
    requires { field.write; }
    param page: Int = 0;
    output result: Rows<Int>;
    compute {
        return if page == 0 { [111, 20013, 73][element()] } else { [73, 111, 0][element()] };
    }
}

kernel b.say@1 {
    requires { field.write; }
    param page: Int = 0;
    output result: Rows<Int>;
    compute {
        return if page == 0 { [65, 66, 0][element()] } else { [90, 0, 0][element()] };
    }
}

param gain: Int = 3;
param page: Int = 0;

input weights: Rows<Int> = \"assets/weights.field\";
input title: Rows<Int> = text \"Hi!\";
input pasted: Rows<Int> = clipboard;

node seed: b.seed@1 { domain = 4; }
node codes: b.codes@1 { page = page; domain = 3; }
node face: font \"fonts/sample.ttf\" { codes <- codes.result; }
node step: b.step@1 {
    prev <~ step.result from seed.result;
    weights <- weights;
    title <- title;
    pasted <- pasted;
    glyphs <- face.glyphs;
    gain = gain;
}
node say: b.say@1 { page = page; domain = 3; }

output result <- step.result;
output copied <- say.result to clipboard;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-bundle-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("assets")).unwrap();
    std::fs::create_dir_all(dir.join("fonts")).unwrap();
    let weights = FieldData::from_storage(
        Schema::new(vec![Scalar::Int]),
        Domain::Rows(3),
        vec![5u32, 7, 11],
    )
    .unwrap();
    std::fs::write(
        dir.join("assets/weights.field"),
        entelechy_formats::file::encode_data(&weights),
    )
    .unwrap();
    std::fs::write(
        dir.join("fonts/sample.ttf"),
        entelechy_import::font::sample::latin(),
    )
    .unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn pack_scene(scene_path: &Path) -> (Session, Vec<u8>) {
    let runtime = Session::runtime_for("jit", Limits::default()).unwrap();
    let mut docs = Session::open_with(scene_path, None, runtime);
    let events = docs.sync();
    assert!(docs.graph().is_some(), "{events:?}");
    let bundle = docs
        .bundle_with_fonts(|_node, path: &Path, static_chars| {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let (compact, _) =
                entelechy_import::font::read_compact(&bytes, static_chars, static_chars.is_some())?;
            Ok(entelechy_session::bundle::BundledAtlas::Compact(compact))
        })
        .expect("scene bundles cleanly");
    let encoded = bundle.encode();
    (docs, encoded)
}

fn output_words(session: &mut Session, port: &str) -> Vec<u32> {
    let handle = session.output(Some(port)).unwrap();
    session.field(handle).unwrap().values().to_vec()
}

#[test]
fn corrupt_or_truncated_bundles_are_refused() {
    let dir = scratch("corrupt");
    let (_, bytes) = pack_scene(&dir.join("scene.esn"));
    let _ = std::fs::remove_dir_all(&dir);

    assert!(Bundle::decode(&[]).is_err());
    assert!(Bundle::decode(b"NOPE\x01\x00\x00\x00").is_err());
    assert!(Bundle::decode(b"ENTB\x02").is_err());
    for cut in [5, bytes.len() / 3, bytes.len() / 2, bytes.len() - 1] {
        assert!(
            Bundle::decode(&bytes[..cut]).is_err(),
            "truncation at {cut} of {} must fail",
            bytes.len()
        );
    }
}

#[test]
fn bundled_session_matches_document_session_across_frames() {
    let dir = scratch("parity");
    let (mut docs, bytes) = pack_scene(&dir.join("scene.esn"));

    let decoded = Bundle::decode(&bytes).expect("bundle decodes");
    assert_eq!(
        decoded.encode(),
        bytes,
        "encode -> decode -> encode is exact"
    );

    let loaded = Session::open_bundle(decoded, "jit", Limits::default(), Wait::blocking(), false)
        .expect("bundle opens on jit");
    assert!(
        loaded.said.iter().any(|s| s == "scene loaded"),
        "{:?}",
        loaded.said
    );
    let mut bundled = loaded.session;

    for frame in 0..5 {
        if frame == 2 {
            docs.set_param("page", Value::Int(1)).unwrap();
            bundled.set_param("page", Value::Int(1)).unwrap();
            docs.set_param("gain", Value::Int(5)).unwrap();
            bundled.set_param("gain", Value::Int(5)).unwrap();
        }
        if frame == 3 {
            docs.place_clipboard("ok!").unwrap();
            bundled.place_clipboard("ok!").unwrap();
        }
        docs.sync_lines();
        bundled.sync_lines();
        let ctx = Context {
            time: f64::from(frame) * 0.016,
            window: [64.0, 48.0],
            ..Context::default()
        };
        docs.evaluate(&ctx).unwrap();
        bundled.evaluate(&ctx).unwrap();
        assert_eq!(
            output_words(&mut bundled, "result"),
            output_words(&mut docs, "result"),
            "frame {frame} differed"
        );
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn turn_based_bundle_prepares_and_reports_scene_loaded() {
    let dir = scratch("turns");
    let (_, bytes) = pack_scene(&dir.join("scene.esn"));
    let _ = std::fs::remove_dir_all(&dir);

    let decoded = Bundle::decode(&bytes).expect("bundle decodes");
    let loaded = Session::open_bundle(decoded, "jit", Limits::default(), Wait::blocking(), true)
        .expect("bundle starts in turns");
    let mut said = loaded.said;
    let mut session = loaded.session;
    for _ in 0..64 {
        if session.graph().is_some() {
            break;
        }
        said.extend(session.sync_turn_lines());
    }
    assert!(session.graph().is_some(), "graph landed after turns");
    assert!(said.iter().any(|s| s == "scene loaded"), "{said:?}");
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(output_words(&mut session, "result").len(), 4);
}

#[test]
fn voxel_example_bundles_and_renders_identically_on_jit() {
    let voxel = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/voxel/scene.esn");
    let (mut docs, bytes) = pack_scene(&voxel);
    let decoded = Bundle::decode(&bytes).expect("voxel bundle decodes");
    let mut bundled =
        Session::open_bundle(decoded, "jit", Limits::default(), Wait::blocking(), false)
            .expect("voxel bundle opens")
            .session;

    let ctx = Context {
        time: 0.0,
        window: [80.0, 60.0],
        scale: 1.0,
        ..Context::default()
    };
    docs.sync_lines();
    bundled.sync_lines();
    docs.evaluate(&ctx).unwrap();
    bundled.evaluate(&ctx).unwrap();
    assert_eq!(
        output_words(&mut bundled, "result"),
        output_words(&mut docs, "result")
    );
}

#[test]
fn v1_bundle_stream_still_decodes_and_bundled_saves_round_trip_across_sessions() {
    let dir = scratch("saves");
    std::fs::create_dir_all(&dir).unwrap();
    let scene_path = dir.join("scene.esn");
    std::fs::write(
        &scene_path,
        "kernel s.fresh@1 { requires { field.write; } output result: Rows<Int>; compute { return 10; } }\n\
         kernel s.bump@1 {\n\
           requires { field.read; field.write; }\n\
           input last: Rows<Int>;\n\
           output result: Rows<Int>;\n\
           compute { return at(last, 0) + 1; }\n\
         }\n\
         input saved: Rows<Int> = \"saves/tally.field\" else fresh.result;\n\
         node fresh: s.fresh@1 { domain = 1; }\n\
         node step: s.bump@1 { last <~ step.result from saved; domain = 1; }\n\
         output result <- step.result;\n\
         output kept <- step.result to \"saves/tally.field\";\n",
    )
    .unwrap();

    let built = Session::build("cpu", Limits::default(), Wait::blocking()).unwrap();
    let mut doc_session = Session::open_with(&scene_path, None, built.runtime);
    doc_session.start_from_nothing_saved();
    doc_session.sync();
    let bundle = doc_session
        .bundle_with_fonts(|_, _, _| unreachable!("no fonts"))
        .unwrap();
    assert_eq!(bundle.saves.len(), 1);
    assert_eq!(bundle.saves[0].slot, "saves/tally.field");
    assert_eq!(bundle.saves[0].output, "kept");
    assert_eq!(bundle.inputs.len(), 0);

    let encoded_v2 = bundle.encode();
    let decoded = Bundle::decode(&encoded_v2).expect("v2 bundle decodes");
    assert_eq!(decoded.saves, bundle.saves);

    // Synthesize a legacy v1 stream (version byte 1 at offset 16, no trailing `saves` list).
    let mut no_saves = bundle.clone();
    no_saves.saves.clear();
    let mut legacy_v1 = no_saves.encode();
    assert_eq!(legacy_v1[16], 2);
    legacy_v1[16] = 1;
    assert_eq!(
        *legacy_v1.last().unwrap(),
        0,
        "empty saves list is 1 zero byte"
    );
    legacy_v1.pop();
    let decoded_v1 = Bundle::decode(&legacy_v1).expect("legacy v1 bundle decodes");
    assert!(decoded_v1.saves.is_empty());

    // First bundled session starts from fallback (10 -> 11 -> 12) and emits dirty save bytes.
    let mut run1 = Session::open_bundle(
        decoded.clone(),
        "jit",
        Limits::default(),
        Wait::blocking(),
        false,
    )
    .unwrap()
    .session;
    assert_eq!(run1.save_slots(), vec!["saves/tally.field"]);
    run1.evaluate(&Context::default()).unwrap();
    run1.confirm().unwrap();
    assert_eq!(output_words(&mut run1, "result"), vec![11]);
    run1.evaluate(&Context::default()).unwrap();
    run1.confirm().unwrap();
    assert_eq!(output_words(&mut run1, "result"), vec![12]);
    let saved_bytes = run1
        .take_dirty_save(0)
        .expect("confirmed frame produced save bytes");
    assert!(
        run1.take_dirty_save(0).is_none(),
        "taken once until next change"
    );

    // Second bundled session preloads `saved_bytes` and continues from 12 -> 13!
    let mut run2 = Session::open_bundle(decoded, "jit", Limits::default(), Wait::blocking(), false)
        .unwrap()
        .session;
    run2.load_save("saves/tally.field", &saved_bytes).unwrap();
    run2.evaluate(&Context::default()).unwrap();
    run2.confirm().unwrap();
    assert_eq!(output_words(&mut run2, "result"), vec![13]);

    let _ = std::fs::remove_dir_all(&dir);
}
