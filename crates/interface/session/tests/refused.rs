//! A node binding a Kernel its document could not publish says why the
//! document was refused, where the fault is written; published again, the
//! Kernel runs and the reason is gone.

use orla_kernel::{Context, Runtime};
use orla_session::Session;

const GOOD: &str = "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { let t = 1; return t; } }\n";
const BAD: &str = "kernel t.k@1 { requires { field.write; } output result: Rows<Int>; compute { let t = 1; let t = 2; return t; } }\n";

#[test]
fn an_unpublished_kernel_says_why_its_document_was_refused() {
    let dir = std::env::temp_dir().join(format!("orla-refused-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), BAD).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"k.orla\";\nnode n: t.k@1 { domain = 1; }\noutput result <- n.result;\n",
    )
    .unwrap();
    let runtime = Runtime::with_backends(1 << 20, Box::new(orla_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    session.sync();
    let said = session
        .evaluate(&Context::default())
        .unwrap_err()
        .to_string();
    assert!(
        said.contains("binds `t.k@1`, which is not published: its document was refused:"),
        "{said}"
    );
    assert!(said.contains("k.orla:1:"), "{said}");
    assert!(said.contains("already bound"), "{said}");

    std::fs::write(dir.join("k.orla"), GOOD).unwrap();
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    std::fs::write(dir.join("k.orla"), BAD).unwrap();
    session.sync();
    // Refused again while running: the running Kernel goes on, unrefused.
    session.evaluate(&Context::default()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}
