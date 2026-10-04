//! A data table input (`.csv`, `.json`) is read as the Kernel reading it
//! lays out its element: a row a record, each number by its name. It is
//! read again when the file changes or the layout does, and one that
//! cannot be read leaves the input as it was and says where.

use std::path::{Path, PathBuf};

use orla_kernel::Context;
use orla_session::{Event, Session};

const KERNELS: &str = "
struct Skill { id: Int; power: Float; at: Vec2; on: Bool; }
kernel t.score@1 {
    requires { field.read; field.write; }
    input skills: Rows<Skill>;
    output result: Rows<Float>;
    compute { let s = sample(skills); return if s.on { float(s.id) * 100.0 + s.power + s.at.y } else { -1.0 }; }
}
";

fn scratch(name: &str, table: &str, file: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-tables-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("data")).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("data").join(file), table).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        format!(
            "use \"k.orla\";\ninput skills: Rows<Skill> = \"data/{file}\";\nnode score: t.score@1 {{ skills <- skills; }}\noutput result <- score.result;\n"
        ),
    )
    .unwrap();
    dir
}

fn open(dir: &Path) -> (Session, Vec<Event>) {
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    (session, events)
}

fn scores(session: &mut Session) -> Vec<f32> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    session
        .field(handle)
        .unwrap()
        .values()
        .iter()
        .map(|w| f32::from_bits(*w))
        .collect()
}

fn rejected(events: &[Event]) -> Vec<String> {
    events
        .iter()
        .filter_map(|event| match event {
            Event::Rejected { error, .. } => Some(error.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_csv_table_is_read_as_its_kernel_lays_it_out_and_read_again() {
    let dir = scratch(
        "csv",
        "id,power,at.x,at.y,on\n3,1.5,0,2,true\n4,9,0,0,false\n",
        "skills.csv",
    );
    let (mut session, events) = open(&dir);
    assert!(rejected(&events).is_empty(), "{events:?}");
    assert_eq!(scores(&mut session), [303.5, -1.0]);
    // Changed, it is read again: a row more.
    std::fs::write(
        dir.join("data/skills.csv"),
        "id,power,at.x,at.y,on\n3,1.5,0,2,true\n4,9,0,0,false\n5,0.25,1,1,1\n",
    )
    .unwrap();
    session.sync();
    assert_eq!(scores(&mut session), [303.5, -1.0, 501.25]);
    // A cell that is not its number: refused, where it is said, and the
    // rows read before stay.
    std::fs::write(
        dir.join("data/skills.csv"),
        "id,power,at.x,at.y,on\n3,1.5,0,2,true\nfour,9,0,0,false\n",
    )
    .unwrap();
    let why = rejected(&session.sync());
    assert!(
        why.iter()
            .any(|w| w.contains("line 3: `id` is \"four\", not an Int")),
        "{why:?}"
    );
    assert_eq!(scores(&mut session), [303.5, -1.0, 501.25]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_json_table_reads_as_the_csv_does() {
    let dir = scratch(
        "json",
        r#"[{"id": 3, "power": 1.5, "at": {"x": 0, "y": 2}, "on": true}, {"id": 4, "power": 9, "at": [0, 0], "on": false}]"#,
        "skills.json",
    );
    let (mut session, events) = open(&dir);
    assert!(rejected(&events).is_empty(), "{events:?}");
    assert_eq!(scores(&mut session), [303.5, -1.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The table follows the Kernel's layout: a field added to the struct is
/// asked of the table at once, though the table did not change.
#[test]
fn a_table_is_read_again_when_its_layout_changes() {
    let dir = scratch(
        "layout",
        "id,power,at.x,at.y,on\n3,1.5,0,2,true\n",
        "skills.csv",
    );
    let (mut session, _) = open(&dir);
    assert_eq!(scores(&mut session), [303.5]);
    std::fs::write(
        dir.join("k.orla"),
        KERNELS.replace("on: Bool; }", "on: Bool; rank: Int; }"),
    )
    .unwrap();
    let why = rejected(&session.sync());
    assert!(
        why.iter().any(|w| w.contains("line 2: no `rank` given")),
        "{why:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A table no Kernel reads has no element to be read as, and says so.
#[test]
fn a_table_no_kernel_reads_is_refused() {
    let dir = scratch("unread", "id\n1\n", "skills.csv");
    std::fs::write(
        dir.join("scene.osn"),
        "use \"k.orla\";\ninput skills: Rows<Skill> = \"data/skills.csv\";\ninput other: Rows<Skill> = \"data/skills.csv\";\nnode score: t.score@1 { skills <- skills; }\noutput result <- score.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("data/skills.csv"),
        "id,power,at.x,at.y,on\n1,0,0,0,true\n",
    )
    .unwrap();
    let (_, events) = open(&dir);
    let why = rejected(&events);
    assert!(
        why.iter()
            .any(|w| w.contains("no published Kernel reads `other`")),
        "{why:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}
