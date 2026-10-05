//! A Kernel that carries many words through its branches is said to be
//! large, with the structs it carries and the functions that carry most;
//! one of many instructions and few words carried is not. A scene that
//! publishes a Kernel no node uses is told so.

/// A Kernel's own struct of `n` floats, and its function passing it
/// through `branches` branches.
fn wide(n: usize, branches: usize) -> String {
    let fields: String = (0..n).map(|i| format!("f{i}: Float; ")).collect();
    let steps: String = (0..branches)
        .map(|i| {
            format!(
                "    let w{} = if x > {i}.5 {{ Wide {{ ..w{i}, f{}: x }} }} else {{ w{i} }};\n",
                i + 1,
                i % n
            )
        })
        .collect();
    let zero: Vec<String> = (0..n).map(|i| format!("f{i}: 0.0")).collect();
    format!(
        "struct Wide {{ {fields}}}
fn widen(x: Float) -> Wide {{
    let w0 = Wide {{ {} }};
{steps}    return w{branches};
}}
",
        zero.join(", ")
    )
}

#[test]
fn a_kernel_carrying_a_wide_struct_through_many_branches_is_large() {
    let source = format!(
        "kernel t.k@1 {{ {} requires {{ field.write; }} output result: Rows<Float>; compute {{ return widen(float(element())).f0; }} }}",
        wide(100, 60)
    );
    let module = orla_compiler::compile(&source).unwrap();
    let said = orla_session::large(&module).expect("6000 words carried");
    assert!(said.contains("`t.k@1` carries 6000 words"), "{said}");
    assert!(said.contains("`Wide` (100 words) 60 times"), "{said}");
    assert!(
        said.contains("`widen` (6000, called by the Kernel itself)"),
        "{said}"
    );
    // Called twice, it is written out twice, and carries twice the words.
    let source = format!(
        "kernel t.k@1 {{ {} requires {{ field.write; }} output result: Rows<Float>; compute {{ let x = float(element()); return widen(x).f0 + widen(x + 1.0).f1; }} }}",
        wide(100, 60)
    );
    let said = orla_session::large(&orla_compiler::compile(&source).unwrap()).unwrap();
    assert!(said.contains("carries 12000 words"), "{said}");
    // The same struct through fewer branches is not.
    let source = format!(
        "kernel t.k@1 {{ {} requires {{ field.write; }} output result: Rows<Float>; compute {{ return widen(float(element())).f0; }} }}",
        wide(100, 20)
    );
    assert!(orla_session::large(&orla_compiler::compile(&source).unwrap()).is_none());
}

#[test]
fn many_instructions_carrying_little_are_not_large() {
    // Some thirty thousand instructions, every value a float.
    let steps: String = (1..10_000)
        .map(|i| format!("    let a{i} = sin(a{}) * {i}.0 + x;\n", i - 1))
        .collect();
    let source = format!(
        "kernel t.k@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{
    let x = float(element());
    let a0 = x;
{steps}    return a9999;
}} }}"
    );
    let module = orla_compiler::compile(&source).unwrap();
    assert!(orla_ir::sizes(&module).total() > 25_000);
    assert!(orla_session::large(&module).is_none());
}

#[test]
fn a_kernel_the_scene_publishes_and_no_node_uses_is_said() {
    let dir = std::env::temp_dir().join(format!("orla-idle-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("lib.orla"),
        "kernel t.used@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }
kernel t.idle@1 { requires { field.write; } output result: Rows<Float>; compute { return 2.0; } }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"lib.orla\";\nnode a: t.used@1 { domain = 1; }\noutput result <- a.result;\n",
    )
    .unwrap();
    let checked = orla_session::check_document(
        &orla_session::Disk::default(),
        &dir.join("scene.osn"),
        false,
    );
    assert_eq!(checked.problems(), 0, "{:?}", checked.lines);
    let idle: Vec<&String> = checked
        .lines
        .iter()
        .map(|(_, line)| line)
        .filter(|line| line.contains("no node of the scene uses"))
        .collect();
    assert_eq!(idle.len(), 1, "{:?}", checked.lines);
    assert!(
        idle[0].contains("lib.orla: publishes `t.idle@1`,"),
        "{}",
        idle[0]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_session_says_a_large_kernel_as_it_publishes_and_idle_ones_once() {
    let dir = std::env::temp_dir().join(format!("orla-shape-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let lib = |extra: &str| {
        format!(
            "kernel t.wide@1 {{ {} requires {{ field.write; }} output result: Rows<Float>; compute {{ return widen(float(element())).f0{extra}; }} }}
kernel t.idle@1 {{ requires {{ field.write; }} output result: Rows<Float>; compute {{ return 2.0; }} }}
",
            wide(100, 60)
        )
    };
    std::fs::write(dir.join("lib.orla"), lib("")).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "use \"lib.orla\";\nnode a: t.wide@1 { domain = 1; }\noutput result <- a.result;\n",
    )
    .unwrap();
    let runtime =
        orla_kernel::Runtime::with_backends(1 << 20, Box::new(orla_cpu::CpuBackend), None);
    let mut session = orla_session::Session::open_with(&dir.join("scene.osn"), None, runtime);
    let said = |events: &[orla_session::Event]| {
        let large: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                orla_session::Event::Large { id, .. } => Some(id.to_string()),
                _ => None,
            })
            .collect();
        let idle: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                orla_session::Event::Idle { ids, .. } => Some(
                    ids.iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join(" "),
                ),
                _ => None,
            })
            .collect();
        (large, idle)
    };
    let (large, idle) = said(&session.sync());
    assert_eq!(large, ["t.wide@1"]);
    assert_eq!(idle, ["t.idle@1"]);
    // Nothing changed: nothing said.
    let (large, idle) = said(&session.sync());
    assert!(large.is_empty() && idle.is_empty(), "{large:?} {idle:?}");
    // A new version, large still: said again; the idle ones as they were.
    std::fs::write(dir.join("lib.orla"), lib(" + 1.0")).unwrap();
    let (large, idle) = said(&session.sync());
    assert_eq!(large, ["t.wide@1"]);
    assert!(idle.is_empty(), "{idle:?}");
    let _ = std::fs::remove_dir_all(&dir);
}
