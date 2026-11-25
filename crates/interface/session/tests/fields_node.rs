//! A `fields` node gives the Field files its `paths` named when last
//! evaluated, their elements one after another in the order named in its
//! `result`, and where each path's are in its `ranges` (-1 and 0 for a
//! path whose file is not in); a file no longer named is let go, one that
//! changes is read again, one of another element refused.

use std::path::{Path, PathBuf};

use entelechy_kernel::{Context, Domain, Scalar, Schema, Value};
use entelechy_session::Session;

/// A Kernel's array literal of `paths`, each path's bytes ended by a 0,
/// padded with 0 to `n`.
fn spelled(paths: &[&str], n: usize) -> String {
    let mut words: Vec<String> = paths
        .iter()
        .flat_map(|path| {
            path.bytes()
                .map(|b| b.to_string())
                .chain(std::iter::once("0".into()))
        })
        .collect();
    words.resize(n, "0".into());
    format!("[{}]", words.join(", "))
}

fn scene(first: &[&str], then: &[&str]) -> String {
    let n = 64;
    format!(
        "struct Spot {{ at: Vec2; id: Int; }}
kernel t.paths@1 {{ requires {{ field.write; }} param which: Int = 0; output result: Rows<Int>;
    compute {{ return if which == 0 {{ {}[element()] }} else {{ {}[element()] }}; }} }}
kernel t.spots@1 {{ requires {{ field.read; field.write; }} input spots: Rows<Spot>; output result: Rows<Spot>; compute {{ return sample(spots); }} }}
kernel t.ranges@1 {{ requires {{ field.read; field.write; }} input ranges: Rows<IVec2>; output result: Rows<IVec2>; compute {{ return sample(ranges); }} }}
param which: Int = 0;
node paths: t.paths@1 {{ which = which; domain = {n}; }}
node near: Rows<Spot> fields \"chunks/\" {{ paths <- paths.result; }}
node spots: t.spots@1 {{ spots <- near.result; }}
node ranges: t.ranges@1 {{ ranges <- near.ranges; }}
output spots <- spots.result;
output ranges <- ranges.result;
",
        spelled(first, n),
        spelled(then, n)
    )
}

/// A chunk of `ids.len()` spots, each at `(id, -id)` with its id.
fn chunk(ids: &[i32]) -> Vec<u8> {
    let schema = Schema::new(vec![Scalar::Float, Scalar::Float, Scalar::Int]);
    let words: Vec<u32> = ids
        .iter()
        .flat_map(|id| [(*id as f32).to_bits(), (-*id as f32).to_bits(), *id as u32])
        .collect();
    entelechy_formats::file::encode(&schema, Domain::Rows(ids.len() as u32), &words)
}

fn scratch(name: &str, first: &[&str], then: &[&str]) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "entelechy-fields-node-{name}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("chunks/far")).unwrap();
    std::fs::write(dir.join("chunks/a.field"), chunk(&[1, 2])).unwrap();
    std::fs::write(dir.join("chunks/b.field"), chunk(&[10, 11, 12])).unwrap();
    std::fs::write(dir.join("chunks/far/c.field"), chunk(&[20])).unwrap();
    // A Field file of another element: one Int a row.
    std::fs::write(
        dir.join("chunks/ints.field"),
        entelechy_formats::file::encode(&Schema::new(vec![Scalar::Int]), Domain::Rows(2), &[7, 8]),
    )
    .unwrap();
    std::fs::write(dir.join("scene.esn"), scene(first, then)).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

fn words(session: &mut Session, output: &str) -> Vec<u32> {
    let handle = session.output(Some(output)).unwrap();
    session.field(handle).unwrap().values().to_vec()
}

/// After an evaluation: each spot's id, and each path's start and count.
fn held(session: &mut Session) -> (Vec<i32>, Vec<(i32, i32)>) {
    session.evaluate(&Context::default()).unwrap();
    let ids = words(session, "spots")
        .chunks(3)
        .map(|spot| spot[2] as i32)
        .collect();
    let ranges = words(session, "ranges")
        .chunks(2)
        .map(|range| (range[0] as i32, range[1] as i32))
        .collect();
    (ids, ranges)
}

#[test]
fn the_files_named_follow_one_another_in_the_order_named_a_frame_later() {
    let dir = scratch(
        "order",
        &["b.field", "none.field", "far/c.field", "a.field"],
        &[],
    );
    let mut session = open(&dir);
    // Before any paths: one zero spot, and no path.
    assert_eq!(held(&mut session), (vec![0], vec![(-1, 0)]));
    // The paths of that evaluation, a frame later; a path naming no file
    // is not in.
    assert_eq!(
        held(&mut session),
        (
            vec![10, 11, 12, 20, 1, 2],
            vec![(0, 3), (-1, 0), (3, 1), (4, 2)]
        )
    );
    // Every element whole: where it is and its id.
    let spots = words(&mut session, "spots");
    assert_eq!(
        (f32::from_bits(spots[0]), f32::from_bits(spots[1])),
        (10.0, -10.0)
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_no_longer_named_is_let_go_and_one_named_again_is_read_again() {
    let dir = scratch("again", &["a.field", "b.field"], &["b.field"]);
    let mut session = open(&dir);
    held(&mut session);
    assert_eq!(held(&mut session).0, vec![1, 2, 10, 11, 12]);
    session.set_param("which", Value::Int(1)).unwrap();
    held(&mut session);
    assert_eq!(held(&mut session), (vec![10, 11, 12], vec![(0, 3)]));
    // Named again, `a.field` is read as it is now.
    std::fs::write(dir.join("chunks/a.field"), chunk(&[5])).unwrap();
    session.set_param("which", Value::Int(0)).unwrap();
    held(&mut session);
    assert_eq!(
        held(&mut session),
        (vec![5, 10, 11, 12], vec![(0, 1), (1, 3)])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_named_is_read_again_when_it_changes() {
    let dir = scratch("changes", &["a.field"], &[]);
    let mut session = open(&dir);
    held(&mut session);
    assert_eq!(held(&mut session).0, vec![1, 2]);
    // A stamp is a modification time and a length: a write lands later.
    std::thread::sleep(std::time::Duration::from_millis(20));
    std::fs::write(dir.join("chunks/a.field"), chunk(&[3, 4, 5])).unwrap();
    session.sync();
    assert_eq!(held(&mut session), (vec![3, 4, 5], vec![(0, 3)]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_file_of_another_element_is_refused_and_said() {
    let dir = scratch("other", &["ints.field", "a.field"], &[]);
    let mut session = open(&dir);
    held(&mut session);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(
        said.iter()
            .any(|line| line.contains("fields node `near`") && line.contains("ints.field")),
        "{said:?}"
    );
    assert_eq!(held(&mut session), (vec![1, 2], vec![(-1, 0), (0, 2)]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_path_out_of_the_directory_is_refused_and_what_was_named_stays() {
    let dir = scratch("out", &["a.field"], &["../scene.esn"]);
    let mut session = open(&dir);
    held(&mut session);
    assert_eq!(held(&mut session).0, vec![1, 2]);
    session.set_param("which", Value::Int(1)).unwrap();
    held(&mut session);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(
        said.iter()
            .any(|line| line.contains("fields node `near`") && line.contains("../scene.esn")),
        "{said:?}"
    );
    assert_eq!(held(&mut session).0, vec![1, 2]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_fields_node_is_said_wrong_where_it_is_written() {
    let dir = scratch("wrong", &[], &[]);
    let said = |text: &str| -> Vec<String> {
        std::fs::write(dir.join("scene.esn"), text).unwrap();
        let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 22);
        session.sync().iter().map(ToString::to_string).collect()
    };
    let none = said("node near: Rows<Float> fields \"chunks/\" { }\noutput o <- near.result;\n");
    assert!(
        none.iter()
            .any(|line| line.contains("a fields node needs `paths <-")),
        "{none:?}"
    );
    let grid = said(
        "kernel t.p@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }\nnode p: t.p@1 { domain = 1; }\nnode near: Grid<Float> fields \"chunks/\" { paths <- p.result; }\noutput o <- near.result;\n",
    );
    assert!(
        grid.iter()
            .any(|line| line.contains("a `Rows` of their element")),
        "{grid:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn off_the_frames_a_file_comes_in_by_a_later_sync_and_frames_go_on_meanwhile() {
    let dir = scratch("background", &["b.field", "a.field"], &[]);
    let mut session = open(&dir);
    session.set_background(true);
    held(&mut session);
    // Frames go on, each held as it was, until the reads are in.
    let began = std::time::Instant::now();
    let in_at_last = loop {
        session.sync();
        let now = held(&mut session);
        let all_in = now.1.len() == 2 && now.1.iter().all(|range| range.0 >= 0);
        if all_in || began.elapsed().as_secs() > 10 {
            break now;
        }
    };
    assert_eq!(in_at_last.0, vec![10, 11, 12, 1, 2]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_narrow_element_is_joined_as_the_kernel_keeps_it() {
    let dir = scratch("narrow", &[], &[]);
    std::fs::create_dir_all(dir.join("bytes")).unwrap();
    for (name, ints) in [("a", vec![7u32, 8]), ("b", vec![1, 2, 3])] {
        std::fs::write(
            dir.join(format!("bytes/{name}.field")),
            entelechy_formats::file::encode(
                &Schema::new(vec![Scalar::Int]),
                Domain::Rows(ints.len() as u32),
                &ints,
            ),
        )
        .unwrap();
    }
    std::fs::write(
        dir.join("scene.esn"),
        format!(
            "kernel t.paths@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ return {}[element()]; }} }}
kernel t.ints@1 {{ requires {{ field.read; field.write; }} input ints: Rows<Int as u8>; output result: Rows<Int>; compute {{ return sample(ints); }} }}
node paths: t.paths@1 {{ domain = 16; }}
node near: Rows<Int> fields \"bytes/\" {{ paths <- paths.result; }}
node ints: t.ints@1 {{ ints <- near.result; }}
output ints <- ints.result;
",
            spelled(&["b.field", "a.field"], 16)
        ),
    )
    .unwrap();
    let mut session = open(&dir);
    session.evaluate(&Context::default()).unwrap();
    session.evaluate(&Context::default()).unwrap();
    assert_eq!(words(&mut session, "ints"), vec![1, 2, 3, 7, 8]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn on_a_device_the_files_are_joined_there_and_none_kept_but_the_result() {
    let gpu = match entelechy_gpu::GpuBackend::new() {
        Ok(gpu) => gpu,
        Err(error) => {
            eprintln!("skipping: {error}");
            return;
        }
    };
    let dir = scratch("device", &["b.field", "a.field"], &["b.field"]);
    let mut runtime = entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(gpu));
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.sync();
    held(&mut session);
    assert_eq!(held(&mut session).0, vec![10, 11, 12, 1, 2]);
    let chunks = |session: &Session| -> Vec<(String, u64, u64)> {
        session
            .memory_map()
            .held
            .into_iter()
            .filter(|entry| entry.owner.starts_with("near: "))
            .map(|entry| (entry.owner, entry.bytes.host, entry.bytes.device))
            .collect()
    };
    let result = |session: &Session| {
        session
            .memory_map()
            .held
            .into_iter()
            .find(|entry| entry.owner == "near.result")
            .map(|entry| (entry.bytes.host, entry.bytes.device))
            .expect("the result is held")
    };
    assert_eq!(chunks(&session), vec![], "joined, no file keeps a Field");
    assert_eq!(result(&session), (0, 5 * 12), "joined on the device");
    // Named alone, `b.field`'s elements come from the result before.
    let fields = session.runtime().host().census().fields;
    session.set_param("which", Value::Int(1)).unwrap();
    held(&mut session);
    assert_eq!(held(&mut session), (vec![10, 11, 12], vec![(0, 3)]));
    assert_eq!(result(&session), (0, 3 * 12));
    assert_eq!(chunks(&session), vec![]);
    assert!(session.runtime().host().census().fields <= fields);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_result_read_again_after_a_scene_left_it_holds_the_files_again() {
    let dir = scratch("unread", &["b.field", "a.field"], &[]);
    let mut session = open(&dir);
    held(&mut session);
    assert_eq!(held(&mut session).0, vec![10, 11, 12, 1, 2]);
    // The scene reads only the ranges for a while, then the result again.
    let full = std::fs::read_to_string(dir.join("scene.esn")).unwrap();
    let ranges_only = full
        .replace("node spots: t.spots@1 { spots <- near.result; }\n", "")
        .replace("output spots <- spots.result;\n", "");
    for text in [ranges_only, full] {
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(dir.join("scene.esn"), text).unwrap();
        let events = session.sync();
        assert!(session.graph().is_some(), "{events:?}");
        session.evaluate(&Context::default()).unwrap();
    }
    for _ in 0..3 {
        session.sync();
        session.evaluate(&Context::default()).unwrap();
    }
    assert_eq!(
        held(&mut session),
        (vec![10, 11, 12, 1, 2], vec![(0, 3), (3, 2)])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_directory_named_through_a_package_is_read_inside_that_package() {
    // The chunks only in a package kept elsewhere, the scene naming their
    // directory by the package's name, as an input's file is.
    let dir = scratch("package", &["b.field", "a.field"], &[]);
    let packages = dir.join("packages");
    std::fs::create_dir_all(packages.join("world.entpkg")).unwrap();
    std::fs::write(
        packages.join("world.entpkg/package.esn"),
        "package world@1 { language = 1; ir = 1; }\n",
    )
    .unwrap();
    std::fs::rename(dir.join("chunks"), packages.join("world.entpkg/chunks")).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        scene(&["b.field", "a.field"], &[])
            .replace("fields \"chunks/\"", "fields \"world.entpkg/chunks/\""),
    )
    .unwrap();
    let store = std::sync::Arc::new(entelechy_session::Disk::with_packages(vec![packages]));
    let mut session = Session::open_in(
        store,
        &dir.join("scene.esn"),
        None,
        entelechy_kernel::Runtime::with_backend(1 << 22, Box::new(entelechy_cpu::CpuBackend)),
    );
    // Reads held to the scene's own directories and its packages, as the
    // Player's are.
    session.restrict_reads().unwrap();
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    held(&mut session);
    assert_eq!(
        held(&mut session),
        (vec![10, 11, 12, 1, 2], vec![(0, 3), (3, 2)])
    );
    let _ = std::fs::remove_dir_all(&dir);
}

fn solid_bc_bytes(
    fmt: entelechy_kernel::BlockFormat,
    w: u32,
    h: u32,
    rgba: [u8; 4],
    ktx2: bool,
) -> Vec<u8> {
    let ch = fmt.channels();
    let mut pixels = vec![0u8; (w as usize) * (h as usize) * ch];
    for p in pixels.chunks_exact_mut(ch) {
        p.copy_from_slice(&rgba[..ch]);
    }
    let blocks =
        entelechy_kernel::CompressedBlocks::encode_unorm8(fmt, Domain::Grid(w, h), &pixels)
            .unwrap();
    if ktx2 {
        entelechy_formats::texture::encode_ktx2(Domain::Grid(w, h), &blocks).unwrap()
    } else {
        entelechy_formats::texture::encode_dds(Domain::Grid(w, h), &blocks).unwrap()
    }
}

#[test]
fn a_volume_fields_node_streams_compressed_ktx2_and_dds_textures_into_a_gpu_array() {
    let dir =
        std::env::temp_dir().join(format!("entelechy-fields-volume-bc-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("textures")).unwrap();
    let fmt = entelechy_kernel::BlockFormat::Bc7RgbaUnorm;
    std::fs::write(
        dir.join("textures/red.ktx2"),
        solid_bc_bytes(fmt, 16, 12, [230, 30, 20, 255], true),
    )
    .unwrap();
    std::fs::write(
        dir.join("textures/green.dds"),
        solid_bc_bytes(fmt, 16, 12, [25, 220, 40, 255], false),
    )
    .unwrap();
    std::fs::write(
        dir.join("textures/wrong_size.ktx2"),
        solid_bc_bytes(fmt, 8, 8, [0, 0, 255, 255], true),
    )
    .unwrap();

    let n = 80;
    let scene_src = format!(
        "kernel t.paths@1 {{
            requires {{ field.write; }}
            param which: Int = 0;
            output result: Rows<Int>;
            compute {{
                return if which == 0 {{ {}[element()] }}
                else if which == 1 {{ {}[element()] }}
                else {{ {}[element()] }};
            }}
        }}
        kernel t.probe@1 {{
            requires {{ field.read; field.write; }}
            input atlas: Volume<Vec4 as unorm8>;
            input ranges: Rows<IVec2>;
            output result: Rows<Vec4>;
            compute {{
                let slot = at(ranges, element());
                return if slot.x < 0 {{
                    vec4(-1.0, -1.0, -1.0, -1.0)
                }} else {{
                    at_linear(atlas, vec3(size(atlas).x * 0.5, size(atlas).y * 0.5, float(slot.x) + 0.5))
                }};
            }}
        }}
        kernel t.ranges@1 {{
            requires {{ field.read; field.write; }}
            input ranges: Rows<IVec2>;
            output result: Rows<IVec2>;
            compute {{ return sample(ranges); }}
        }}
        param which: Int = 0;
        node paths: t.paths@1 {{ which = which; domain = {n}; }}
        node atlas: Volume<Vec4 as unorm8> fields \"textures/\" {{ paths <- paths.result; }}
        node probe: t.probe@1 {{ atlas <- atlas.result; ranges <- atlas.ranges; domain = 3; }}
        node ranges: t.ranges@1 {{ ranges <- atlas.ranges; }}
        output probe <- probe.result;
        output ranges <- ranges.result;
        ",
        spelled(&["red.ktx2", "missing.ktx2", "green.dds"], n),
        spelled(&["green.dds"], n),
        spelled(&["green.dds", "wrong_size.ktx2"], n),
    );
    std::fs::write(dir.join("scene.esn"), scene_src).unwrap();

    let (mut session, is_hw_bc) = match entelechy_gpu::GpuBackend::new() {
        Ok(gpu) => {
            let bc = gpu.supports_texture_compression_bc();
            let mut runtime = entelechy_kernel::Runtime::with_backend(4 << 20, Box::new(gpu));
            runtime.set_small(0);
            (
                Session::open_with(&dir.join("scene.esn"), None, runtime),
                bc,
            )
        }
        Err(_) => (open(&dir), false),
    };
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");

    // Frame 1 feeds paths; Frame 2 reads the joined compressed Texture2DArray.
    session.evaluate(&Context::default()).unwrap();
    session.evaluate(&Context::default()).unwrap();

    let ranges: Vec<(i32, i32)> = words(&mut session, "ranges")
        .chunks(2)
        .map(|r| (r[0] as i32, r[1] as i32))
        .collect();
    assert_eq!(ranges, vec![(0, 1), (-1, 0), (1, 1)]);

    let probed: Vec<[f32; 4]> = words(&mut session, "probe")
        .chunks(4)
        .map(|p| {
            [
                f32::from_bits(p[0]),
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ]
        })
        .collect();
    assert!((probed[0][0] - 230.0 / 255.0).abs() < 0.03, "{probed:?}");
    assert!((probed[0][1] - 30.0 / 255.0).abs() < 0.03, "{probed:?}");
    assert_eq!(probed[1], [-1.0, -1.0, -1.0, -1.0]);
    assert!((probed[2][0] - 25.0 / 255.0).abs() < 0.03, "{probed:?}");
    assert!((probed[2][1] - 220.0 / 255.0).abs() < 0.03, "{probed:?}");

    let layer_bytes = fmt.byte_len(Domain::Grid(16, 12)).unwrap() as u64;
    if is_hw_bc {
        let result_mem = session
            .memory_map()
            .held
            .into_iter()
            .find(|e| e.owner == "atlas.result")
            .map(|e| (e.bytes.host, e.bytes.device))
            .expect("atlas.result is held");
        assert_eq!(
            result_mem,
            (0, 2 * layer_bytes),
            "2 compressed BC7 layers must occupy exact compressed VRAM with zero Host RAM copy"
        );
    }

    // Evict red.ktx2 -> green.dds compacts GPU-to-GPU down to layer 0!
    session.set_param("which", Value::Int(1)).unwrap();
    session.evaluate(&Context::default()).unwrap();
    session.evaluate(&Context::default()).unwrap();
    let ranges: Vec<(i32, i32)> = words(&mut session, "ranges")
        .chunks(2)
        .map(|r| (r[0] as i32, r[1] as i32))
        .collect();
    assert_eq!(ranges, vec![(0, 1)]);
    let probed: Vec<[f32; 4]> = words(&mut session, "probe")
        .chunks(4)
        .map(|p| {
            [
                f32::from_bits(p[0]),
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ]
        })
        .collect();
    assert!((probed[0][1] - 220.0 / 255.0).abs() < 0.03, "{probed:?}");
    if is_hw_bc {
        let result_mem = session
            .memory_map()
            .held
            .into_iter()
            .find(|e| e.owner == "atlas.result")
            .map(|e| (e.bytes.host, e.bytes.device))
            .expect("atlas.result is held");
        assert_eq!(result_mem, (0, layer_bytes));
    }

    // Adding a mismatched extent texture rejects it cleanly while preserving green.dds.
    session.set_param("which", Value::Int(2)).unwrap();
    session.evaluate(&Context::default()).unwrap();
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(
        said.iter()
            .any(|line| line.contains("wrong_size.ktx2") && line.contains("8x8")),
        "{said:?}"
    );
    session.evaluate(&Context::default()).unwrap();
    let ranges: Vec<(i32, i32)> = words(&mut session, "ranges")
        .chunks(2)
        .map(|r| (r[0] as i32, r[1] as i32))
        .collect();
    assert_eq!(ranges, vec![(0, 1), (-1, 0)]);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn static_ktx2_and_dds_inputs_stay_compressed_on_gpu_and_match_cpu() {
    let dir =
        std::env::temp_dir().join(format!("entelechy-static-bc-inputs-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("albedo.ktx2"),
        solid_bc_bytes(
            entelechy_kernel::BlockFormat::Bc7RgbaUnorm,
            16,
            16,
            [200, 100, 50, 255],
            true,
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("normal.dds"),
        solid_bc_bytes(
            entelechy_kernel::BlockFormat::Bc5RgUnorm,
            16,
            16,
            [128, 192, 0, 0],
            false,
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("rough.ktx2"),
        solid_bc_bytes(
            entelechy_kernel::BlockFormat::Bc4RUnorm,
            16,
            16,
            [77, 0, 0, 0],
            true,
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "kernel t.read@1 {
            requires { field.read; field.write; }
            input albedo: Grid<Vec4 as unorm8>;
            input normal: Grid<Vec2 as unorm8>;
            input rough: Grid<Float as unorm8>;
            output result: Grid<Vec4>;
            compute {
                let p = coord() + vec2(0.5, 0.5);
                let a = at_linear(albedo, p);
                let n = at_linear(normal, p);
                let r = at_linear(rough, p);
                return vec4(a.x, n.y, r, a.w);
            }
        }
        input albedo: Grid<Vec4 as unorm8> = \"albedo.ktx2\";
        input normal: Grid<Vec2 as unorm8> = \"normal.dds\";
        input rough: Grid<Float as unorm8> = \"rough.ktx2\";
        node out: t.read@1 { albedo <- albedo; normal <- normal; rough <- rough; }
        output result <- out.result;
        ",
    )
    .unwrap();

    let mut cpu_session = Session::open_with(
        &dir.join("scene.esn"),
        None,
        entelechy_kernel::Runtime::with_backend(4 << 20, Box::new(entelechy_cpu::CpuBackend)),
    );
    let events = cpu_session.sync();
    assert!(cpu_session.graph().is_some(), "{events:?}");
    cpu_session.evaluate(&Context::default()).unwrap();
    let cpu_words = words(&mut cpu_session, "result");

    if let Ok(gpu) = entelechy_gpu::GpuBackend::new() {
        let is_hw_bc = gpu.supports_texture_compression_bc();
        let mut runtime = entelechy_kernel::Runtime::with_backend(4 << 20, Box::new(gpu));
        runtime.set_small(0);
        let mut gpu_session = Session::open_with(&dir.join("scene.esn"), None, runtime);
        let events = gpu_session.sync();
        assert!(gpu_session.graph().is_some(), "{events:?}");
        if is_hw_bc {
            let held = gpu_session.memory_map().held;
            let bytes_of = |name: &str| {
                held.iter()
                    .find(|e| e.owner == name)
                    .map(|e| (e.bytes.host, e.bytes.device))
                    .unwrap_or_else(|| panic!("{name} missing from memory map"))
            };
            assert_eq!(bytes_of("albedo"), (0, 256)); // 16x16 BC7 = 16 blocks * 16 B
            assert_eq!(bytes_of("normal"), (0, 256)); // 16x16 BC5 = 16 blocks * 16 B
            assert_eq!(bytes_of("rough"), (0, 128)); // 16x16 BC4 = 16 blocks * 8 B
        }
        gpu_session.evaluate(&Context::default()).unwrap();
        let gpu_words = words(&mut gpu_session, "result");
        for (idx, (&cw, &gw)) in cpu_words.iter().zip(gpu_words.iter()).enumerate() {
            let cf = f32::from_bits(cw);
            let gf = f32::from_bits(gw);
            assert!(
                (cf - gf).abs() <= (1.0 / 255.0) + 1e-4,
                "word {idx}: cpu={cf} vs gpu={gf}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn background_io_pool_streams_and_cancels_volume_textures_cleanly() {
    let dir = std::env::temp_dir().join(format!("entelechy-bg-volume-pool-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("textures")).unwrap();
    let fmt = entelechy_kernel::BlockFormat::Bc7RgbaUnorm;
    for (name, rgb) in [
        ("t0.ktx2", [240, 20, 20, 255]),
        ("t1.ktx2", [20, 240, 20, 255]),
        ("t2.ktx2", [20, 20, 240, 255]),
    ] {
        std::fs::write(
            dir.join("textures").join(name),
            solid_bc_bytes(fmt, 16, 16, rgb, true),
        )
        .unwrap();
    }
    let n = 64;
    let scene_src = format!(
        "kernel t.paths@1 {{
            requires {{ field.write; }}
            param which: Int = 0;
            output result: Rows<Int>;
            compute {{ return if which == 0 {{ {}[element()] }} else {{ {}[element()] }}; }}
        }}
        kernel t.ranges@1 {{
            requires {{ field.read; field.write; }}
            input ranges: Rows<IVec2>;
            output result: Rows<IVec2>;
            compute {{ return sample(ranges); }}
        }}
        kernel t.probe@1 {{
            requires {{ field.read; field.write; }}
            input atlas: Volume<Vec4 as unorm8>;
            output result: Rows<Vec4>;
            compute {{ return at(atlas, vec3(4.0, 4.0, float(element()))); }}
        }}
        param which: Int = 0;
        node paths: t.paths@1 {{ which = which; domain = {n}; }}
        node atlas: Volume<Vec4 as unorm8> fields \"textures/\" {{ paths <- paths.result; }}
        node ranges: t.ranges@1 {{ ranges <- atlas.ranges; }}
        node probe: t.probe@1 {{ atlas <- atlas.result; domain = 2; }}
        output ranges <- ranges.result;
        output probe <- probe.result;
        ",
        spelled(&["t0.ktx2", "t1.ktx2", "t2.ktx2"], n),
        spelled(&["t1.ktx2", "t2.ktx2"], n),
    );
    std::fs::write(dir.join("scene.esn"), scene_src).unwrap();

    let mut session = open(&dir);
    session.set_background(true);
    // Request t0/t1/t2 in background, then immediately swap to t1/t2 (cancelling t0).
    session.evaluate(&Context::default()).unwrap();
    session.set_param("which", Value::Int(1)).unwrap();
    session.evaluate(&Context::default()).unwrap();

    let began = std::time::Instant::now();
    loop {
        session.sync();
        session.evaluate(&Context::default()).unwrap();
        let ranges: Vec<(i32, i32)> = words(&mut session, "ranges")
            .chunks(2)
            .map(|r| (r[0] as i32, r[1] as i32))
            .collect();
        if ranges == vec![(0, 1), (1, 1)] {
            break;
        }
        assert!(
            began.elapsed().as_secs() < 10,
            "background streaming timed out: {ranges:?}"
        );
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    let probed: Vec<[f32; 4]> = words(&mut session, "probe")
        .chunks(4)
        .map(|p| {
            [
                f32::from_bits(p[0]),
                f32::from_bits(p[1]),
                f32::from_bits(p[2]),
                f32::from_bits(p[3]),
            ]
        })
        .collect();
    assert!((probed[0][1] - 240.0 / 255.0).abs() < 0.03, "{probed:?}");
    assert!((probed[1][2] - 240.0 / 255.0).abs() < 0.03, "{probed:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_volume_fields_node_streams_multimip_ktx2_and_dds_with_trilinear_and_aniso() {
    let dir = std::env::temp_dir().join(format!(
        "entelechy-fields-volume-multimip-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("textures")).unwrap();
    let fmt = entelechy_kernel::BlockFormat::Bc7RgbaUnorm;
    let w = 32u32;
    let h = 32u32;
    let mut stripe_px = vec![0u8; (w * h * 4) as usize];
    for y in 0..h as usize {
        for x in 0..w as usize {
            let b = (y * w as usize + x) * 4;
            let v = if y % 2 == 0 { 240u8 } else { 16u8 };
            stripe_px[b..b + 4].copy_from_slice(&[v, v, v, 255]);
        }
    }
    let blocks_ktx = entelechy_kernel::CompressedBlocks::encode_unorm8_mips(
        fmt,
        Domain::Grid(w, h),
        &stripe_px,
        6,
        false,
    )
    .unwrap();
    let mips = blocks_ktx.mip_count();
    assert_eq!(mips, 6);
    std::fs::write(
        dir.join("textures/stripes.ktx2"),
        entelechy_formats::texture::encode_ktx2(Domain::Grid(w, h), &blocks_ktx).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.join("textures/stripes.dds"),
        entelechy_formats::texture::encode_dds(Domain::Grid(w, h), &blocks_ktx).unwrap(),
    )
    .unwrap();

    let n = 64;
    let scene_src = format!(
        "kernel t.paths@1 {{
            requires {{ field.write; }}
            output result: Rows<Int>;
            compute {{ return {}[element()]; }}
        }}
        kernel t.probe@1 {{
            requires {{ field.read; field.write; }}
            input atlas: Volume<Vec4 as unorm8>;
            output result: Rows<Vec4>;
            compute {{
                let layer = float(element()) + 0.5;
                let iso = at_repeat(atlas, vec3(0.5, 0.5, layer), 3.0);
                let aniso_hi = at_repeat(atlas, vec3(0.25, 0.5, layer), vec2(10.0, 0.0), vec2(0.0, 0.25));
                let aniso_lo = at_repeat(atlas, vec3(0.25, 1.5, layer), vec2(10.0, 0.0), vec2(0.0, 0.25));
                return vec4(iso.x, aniso_hi.x, aniso_lo.x, 1.0);
            }}
        }}
        node paths: t.paths@1 {{ domain = {n}; }}
        node atlas: Volume<Vec4 as unorm8> fields \"textures/\" {{ paths <- paths.result; }}
        node probe: t.probe@1 {{ atlas <- atlas.result; domain = 2; }}
        output probe <- probe.result;
        ",
        spelled(&["stripes.ktx2", "stripes.dds"], n),
    );
    std::fs::write(dir.join("scene.esn"), scene_src).unwrap();

    let (mut session, is_hw_bc) = match entelechy_gpu::GpuBackend::new() {
        Ok(gpu) => {
            let bc = gpu.supports_texture_compression_bc();
            let mut runtime = entelechy_kernel::Runtime::with_backend(4 << 20, Box::new(gpu));
            runtime.set_small(0);
            (
                Session::open_with(&dir.join("scene.esn"), None, runtime),
                bc,
            )
        }
        Err(_) => (open(&dir), false),
    };
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session.evaluate(&Context::default()).unwrap();
    session.evaluate(&Context::default()).unwrap();

    if is_hw_bc {
        let expected_bytes = fmt
            .mip_chain_byte_len(Domain::Volume(w, h, 2), mips)
            .unwrap() as u64;
        let result_mem = session
            .memory_map()
            .held
            .into_iter()
            .find(|e| e.owner == "atlas.result")
            .map(|e| (e.bytes.host, e.bytes.device))
            .expect("atlas.result is held");
        assert_eq!(
            result_mem,
            (0, expected_bytes),
            "2 streamed multi-mip BC7 layers must occupy full compressed mip chain in VRAM"
        );
    }

    let raw = words(&mut session, "probe");
    for layer in 0..2 {
        let iso = f32::from_bits(raw[layer * 4]);
        let aniso_hi = f32::from_bits(raw[layer * 4 + 1]);
        let aniso_lo = f32::from_bits(raw[layer * 4 + 2]);
        assert!((iso - 0.5).abs() < 0.08, "layer {layer}: iso={iso}");
        assert!(aniso_hi > 0.78, "layer {layer}: aniso_hi={aniso_hi}");
        assert!(aniso_lo < 0.22, "layer {layer}: aniso_lo={aniso_lo}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}
