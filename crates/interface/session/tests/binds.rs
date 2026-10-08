//! A node's binding that its Kernel's ports alone decide is wrong is said
//! by a check, unrun, in the words an evaluation says it; what only the
//! data decides is left to the evaluation.

use orla_kernel::{Context, Runtime};
use orla_session::Session;

const LIB: &str = "kernel t.fill@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }
kernel t.sized@1 { requires { field.read; field.write; } input fit: Rows<Float>; output result: Rows<Float>; compute { return 2.0; } }
kernel t.scaled@1 { requires { field.write; } param k: Float = 1.0; output result: Rows<Float>; compute { return k; } }
kernel t.plane@1 { requires { field.write; } output result: Grid<Float>; compute { return 1.0; } }
struct Corner { clip: Vec4; normal: Vec3; uv: Vec2; }
struct Tri { a: Int; b: Int; c: Int; }
kernel t.corners@1 { requires { field.write; } output result: Rows<Corner>; compute { return Corner { clip: vec4(0.0, 0.0, 0.5, 1.0), normal: vec3(0.0, 0.0, 1.0), uv: vec2(0.0, 0.0) }; } }
kernel t.tris@1 { requires { field.write; } output result: Rows<Tri>; compute { return Tri { a: 0, b: 1, c: 2 }; } }
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
        (
            "native",
            "node c: t.corners@1 { domain = 3; }\nnode t: t.tris@1 { domain = 1; }\nnode r: native.rasterize@1 { vertices <- c.result; triangles <- t.result; domain = vec2(value(fit, 0), 4); }\noutput result <- r.surface;\n",
            "`value(fit)`: `fit` is not an input of this Kernel",
        ),
        (
            "shape",
            "node a: t.plane@1 { domain = 3; }\noutput result <- a.result;\n",
            "output is a Grid but the domain is a Rows",
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

/// What a check of `scene`, over `lib`, notes and says.
fn checked_lines(name: &str, lib: &str, scene: &str) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("orla-binds-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("lib.orla"), lib).unwrap();
    std::fs::write(dir.join("scene.osn"), format!("use \"lib.orla\";\n{scene}")).unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    let _ = std::fs::remove_dir_all(&dir);
    checked.lines.iter().map(|(_, line)| line.clone()).collect()
}

/// A node with no `domain` that carries its own output back through an
/// input other than its first runs over that first input, so its state
/// takes another's size: a check notes it. Its own `domain`, the carried
/// port first, or a first input made from what it carries says nothing.
#[test]
fn a_check_notes_state_carried_past_the_first_input() {
    let lib = "kernel t.one@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }
kernel t.step@1 { requires { field.read; field.write; } input table: Rows<Float>; input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) + 1.0; } }
kernel t.first@1 { requires { field.read; field.write; } input previous: Rows<Float>; input table: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) + 1.0; } }
kernel t.pass@1 { requires { field.read; field.write; } input from: Rows<Float>; output result: Rows<Float>; compute { return sample(from); } }
";
    let notes = |name: &str, scene: &str| -> Vec<String> {
        checked_lines(name, lib, scene)
            .into_iter()
            .filter(|line| line.contains("carries its own"))
            .collect()
    };
    let head = "node start: t.one@1 { domain = 1; }\nnode table: t.one@1 { domain = 5; }\n";
    let said = notes(
        "carried-past",
        &format!(
            "{head}node s: t.step@1 {{ table <- table.result; previous <~ s.result from start.result; }}\noutput o <- s.result;\n"
        ),
    );
    assert_eq!(said.len(), 1, "{said:?}");
    assert!(
        said[0].contains("node `s` carries its own `result` back through `previous`")
            && said[0].contains("its first input `table`"),
        "{said:?}"
    );
    for (name, scene) in [
        (
            "carried-domain",
            format!(
                "{head}node s: t.step@1 {{ table <- table.result; previous <~ s.result from start.result; domain = 1; }}\noutput o <- s.result;\n"
            ),
        ),
        (
            "carried-first",
            format!(
                "{head}node s: t.first@1 {{ previous <~ s.result from start.result; table <- table.result; }}\noutput o <- s.result;\n"
            ),
        ),
        (
            "carried-made",
            format!(
                "{head}node newer: t.pass@1 {{ from <~ s.result from start.result; }}\nnode s: t.step@1 {{ table <- newer.result; previous <~ s.result from start.result; }}\noutput o <- s.result;\n"
            ),
        ),
    ] {
        let said = notes(name, &scene);
        assert!(said.is_empty(), "{name}: {said:?}");
    }
}
