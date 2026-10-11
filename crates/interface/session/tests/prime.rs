//! The documents a scene uses, their new Kernels compiled and prepared at
//! once: each prepared once, and published, refused and said in the
//! order of the documents, as if each were read, compiled and prepared in
//! turn.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use entelechy_kernel::{Backend, Handle, Invocation, Prepared, Runtime, Scope, Validated};
use entelechy_session::{Event, Session};

/// The interpreter, counting what it prepares.
struct Counting(Arc<AtomicUsize>);

impl Backend for Counting {
    fn name(&self) -> &'static str {
        "counting"
    }
    fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
        self.0.fetch_add(1, Ordering::Relaxed);
        entelechy_cpu::CpuBackend.prepare(module)
    }
    fn execute(
        &self,
        prepared: &dyn Prepared,
        module: &Validated,
        invocation: &Invocation,
        scope: &mut Scope<'_>,
    ) -> Result<Vec<Handle>, entelechy_kernel::ExecutionError> {
        Backend::execute(
            &entelechy_cpu::CpuBackend,
            prepared,
            module,
            invocation,
            scope,
        )
    }
}

fn kernel(id: &str, value: &str) -> String {
    format!(
        "kernel {id} {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return {value}; }} }}\n"
    )
}

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-prime-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let write = |file: &str, text: String| std::fs::write(dir.join(file), text).unwrap();
    write("a.ent", kernel("t.a1@1", "1.0") + &kernel("t.a2@1", "2.0"));
    // One Kernel calls what is nowhere: the document is refused.
    write("b.ent", kernel("t.b1@1", "nowhere()"));
    write("c.ent", kernel("t.c1@1", "3.0") + &kernel("t.c2@1", "4.0"));
    write(
        "scene.esn",
        "use \"a.ent\";\nuse \"b.ent\";\nuse \"c.ent\";\n\
         node a1: t.a1@1 { domain = 1; }\nnode c2: t.c2@1 { domain = 1; }\noutput result <- c2.result;\n"
            .to_owned(),
    );
    dir
}

/// Each event as its kind and the file it names.
fn said(events: &[Event]) -> Vec<(&'static str, String)> {
    let file = |path: &Path| path.file_name().unwrap().to_string_lossy().into_owned();
    events
        .iter()
        .filter_map(|event| match event {
            Event::Published { path, .. } => Some(("published", file(path))),
            Event::Rejected { path, .. } => Some(("rejected", file(path))),
            _ => None,
        })
        .collect()
}

#[test]
fn kernels_prepared_at_once_are_prepared_once_and_said_in_order() {
    let dir = scratch();
    let prepared = Arc::new(AtomicUsize::new(0));
    let runtime = Runtime::with_backends(1 << 20, Box::new(Counting(Arc::clone(&prepared))), None);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    let events = session.sync();
    assert_eq!(
        said(&events),
        [
            ("published", "a.ent".to_owned()),
            ("published", "a.ent".to_owned()),
            ("rejected", "b.ent".to_owned()),
            ("published", "c.ent".to_owned()),
            ("published", "c.ent".to_owned()),
        ],
        "{events:?}"
    );
    // Four Kernels compiled, each prepared once; the refused one never.
    assert_eq!(prepared.load(Ordering::Relaxed), 4);
    // One changed: it alone is prepared again.
    std::fs::write(
        dir.join("c.ent"),
        kernel("t.c1@1", "5.0") + &kernel("t.c2@1", "4.0"),
    )
    .unwrap();
    session.sync();
    session.sync();
    assert_eq!(prepared.load(Ordering::Relaxed), 5);
    let _ = std::fs::remove_dir_all(&dir);
}
