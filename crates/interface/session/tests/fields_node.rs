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
        if now.0 != vec![0] || began.elapsed().as_secs() > 10 {
            break now;
        }
    };
    assert_eq!(in_at_last.0, vec![10, 11, 12, 1, 2]);
    let _ = std::fs::remove_dir_all(&dir);
}
