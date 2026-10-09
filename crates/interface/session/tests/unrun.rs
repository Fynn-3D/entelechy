//! A node with no `domain` runs over its first input, one element each; a
//! Kernel that compares its element with that input's count (a join of
//! two rows) then never runs the elements past it. A check says so, and a
//! session says so once when the scene loads; with its `domain` written,
//! neither does.

use orla_kernel::{Context, Runtime};
use orla_session::{Event, Session};

const JOIN: &str = "kernel t.a@1 { requires { field.write; } output result: Rows<Int>; compute { return 1; } }
kernel t.b@1 { requires { field.write; } output result: Rows<Int>; compute { return 2; } }
kernel t.join@1 { requires { field.read; field.write; } input first: Rows<Int>; input second: Rows<Int>; output result: Rows<Int>;
    compute { let n = count(first); return if element() < n { at(first, element()) } else { at(second, element() - n) }; } }
";

fn scene(domain: &str) -> String {
    format!(
        "use \"k.orla\";\nnode a: t.a@1 {{ domain = 2; }}\nnode b: t.b@1 {{ domain = 3; }}\nnode j: t.join@1 {{ first <- a.result; second <- b.result;{domain} }}\noutput result <- j.result;\n"
    )
}

fn said(name: &str, domain: &str) -> (Vec<String>, Vec<String>, usize) {
    let dir = std::env::temp_dir().join(format!("orla-unrun-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), JOIN).unwrap();
    std::fs::write(dir.join("scene.osn"), scene(domain)).unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    let notes = checked
        .lines
        .iter()
        .filter(|(_, line)| line.contains("compares `element()`"))
        .map(|(_, line)| line.clone())
        .collect();
    let runtime = Runtime::with_backends(1 << 20, Box::new(orla_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    let mut events: Vec<String> = session
        .sync()
        .iter()
        .filter_map(|event| match event {
            Event::Unrun { node, .. } => Some(node.clone()),
            _ => None,
        })
        .collect();
    let report = session.evaluate(&Context::default()).unwrap();
    let count = session.field(report.output).unwrap().storage().len();
    // Said once: a sync with nothing changed says it no more.
    events.extend(session.sync().iter().filter_map(|event| match event {
        Event::Unrun { node, .. } => Some(node.clone()),
        _ => None,
    }));
    let _ = std::fs::remove_dir_all(&dir);
    (notes, events, count)
}

#[test]
fn a_join_with_no_domain_is_said_to_leave_elements_unrun() {
    let (notes, events, count) = said("bare", "");
    assert_eq!(notes.len(), 1, "{notes:?}");
    assert!(notes[0].contains("node `j` has no `domain`"), "{notes:?}");
    assert!(notes[0].contains("`count(first)`"), "{notes:?}");
    assert_eq!(events, vec!["j".to_string()]);
    assert_eq!(count, 2);
}

#[test]
fn a_join_with_its_domain_is_not() {
    let (notes, events, count) = said("written", " domain = count(first) + count(second);");
    assert!(notes.is_empty(), "{notes:?}");
    assert!(events.is_empty(), "{events:?}");
    assert_eq!(count, 5);
}
