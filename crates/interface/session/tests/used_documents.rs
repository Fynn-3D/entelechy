//! A library is read again when a document it uses changes, however many
//! syncs found nothing changed before: a sync that finds every used
//! document as the libraries were read with them keeps those documents,
//! each once, and asks of them alone until one changes.

use entelechy_session::Session;

fn value(session: &mut Session) -> f32 {
    let report = session
        .evaluate(&entelechy_kernel::Context::default())
        .unwrap();
    f32::from_bits(session.runtime().host().element(report.output, 0).unwrap()[0])
}

#[test]
fn a_change_to_a_used_document_is_found_after_syncs_that_found_none() {
    let dir = std::env::temp_dir().join(format!("entelechy-used-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"made.ent\";\nnode made: t.made@1 { domain = 1; }\noutput result <- made.result;\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("made.ent"),
        "use \"base.ent\";\nkernel t.made@1 { requires { field.write; } output result: Rows<Float>; compute { return base(); } }\n",
    )
    .unwrap();
    let base = |text: &str| {
        std::fs::write(
            dir.join("base.ent"),
            format!("fn base() -> Float {{ return {text}; }}\n"),
        )
        .unwrap();
    };
    base("1.0");
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.sync();
    assert_eq!(value(&mut session), 1.0);
    // Syncs that find nothing, then a change to the document `made.ent`
    // uses alone, of another length so its stamp tells it apart.
    for (text, expected) in [("22.0", 22.0), ("333.0", 333.0)] {
        assert!(session.sync().is_empty());
        assert!(session.sync().is_empty());
        base(text);
        assert!(!session.sync().is_empty());
        assert_eq!(value(&mut session), expected);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
