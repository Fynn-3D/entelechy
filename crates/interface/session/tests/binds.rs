//! A node's binding that its Kernel's ports alone decide is wrong is said
//! by a check, unrun, in the words an evaluation says it; what only the
//! data decides is left to the evaluation.

use orla_kernel::{Context, Runtime};
use orla_session::Session;

const LIB: &str = "kernel t.fill@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }
kernel t.sized@1 { requires { field.read; field.write; } input fit: Rows<Float>; output result: Rows<Float>; compute { return 2.0; } }
kernel t.scaled@1 { requires { field.write; } param k: Float = 1.0; output result: Rows<Float>; compute { return k; } }
";

/// What a check of `scene` says is a problem, and what evaluating it says.
fn said(name: &str, scene: &str) -> (Vec<String>, String) {
    let dir = std::env::temp_dir().join(format!("orla-binds-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.orla"), LIB).unwrap();
    std::fs::write(dir.join("scene.osn"), format!("use \"lib.orla\";\n{scene}")).unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    let problems = checked
        .lines
        .iter()
        .filter(|(problem, _)| *problem)
        .map(|(_, line)| line.clone())
        .collect();
    let runtime = Runtime::with_backends(1 << 20, Box::new(orla_cpu::CpuBackend), None);
    let mut session = Session::open_with(&dir.join("scene.osn"), None, runtime);
    session.sync();
    let evaluated = match session.evaluate(&Context::default()) {
        Ok(_) => String::new(),
        Err(error) => error.to_string(),
    };
    let _ = std::fs::remove_dir_all(&dir);
    (problems, evaluated)
}

#[test]
fn a_binding_its_kernel_refuses_is_said_by_the_check_as_the_evaluation_says_it() {
    for (name, scene, fault) in [
        (
            "value",
            "node a: t.fill@1 { domain = value(fit, 0); }\noutput result <- a.result;\n",
            "`value(fit)`: `fit` is not an input of this Kernel",
        ),
        (
            "port",
            "node s: t.fill@1 { domain = 2; }\nnode a: t.fill@1 { nope <- s.result; domain = 3; }\noutput result <- a.result;\n",
            "node `a` refers to `nope`, which its kernel does not declare",
        ),
        (
            "unbound",
            "node a: t.sized@1 { domain = 3; }\noutput result <- a.result;\n",
            "node `a` does not bind input `fit`",
        ),
        (
            "param",
            "node a: t.scaled@1 { k = vec2(1.0, 2.0); domain = 3; }\noutput result <- a.result;\n",
            "node `a` sets `k` to",
        ),
    ] {
        let (problems, evaluated) = said(name, scene);
        assert!(
            problems.iter().any(|line| line.contains(fault)),
            "{name}: the check says {problems:?}"
        );
        assert!(
            evaluated.contains(fault),
            "{name}: the evaluation says {evaluated}"
        );
    }
}

#[test]
fn what_only_the_data_decides_is_left_to_the_evaluation() {
    // `fit` has three elements; the sixth is asked for.
    let (problems, evaluated) = said(
        "data",
        "node a: t.fill@1 { domain = 3; }\nnode w: t.sized@1 { fit <- a.result; domain = value(fit, 5); }\noutput result <- w.result;\n",
    );
    assert!(problems.is_empty(), "{problems:?}");
    assert!(evaluated.contains("`fit` has 3 element(s)"), "{evaluated}");
    // Read within it, it holds.
    let (problems, evaluated) = said(
        "fits",
        "node a: t.fill@1 { domain = 3; }\nnode w: t.sized@1 { fit <- a.result; domain = value(fit, 0); }\noutput result <- w.result;\n",
    );
    assert!(problems.is_empty(), "{problems:?}");
    assert!(evaluated.is_empty(), "{evaluated}");
}
