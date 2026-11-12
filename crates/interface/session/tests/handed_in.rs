//! Files handed in (a page's): a package named alone is found in
//! `packages/` among them, as one beside the program is on a disk.

use std::path::Path;
use std::sync::Arc;

use entelechy_session::{Limits, Memory, Session, Store, Wait};

#[test]
fn a_package_named_alone_is_found_among_files_handed_in() {
    let files = Memory::default();
    let put = |path: &str, text: &str| files.write(Path::new(path), text.as_bytes()).expect("kept");
    put(
        "packages/ones.entpkg/package.esn",
        "package ones@1 { language = 1; ir = 1; }\n\nuse \"kernels/\";\n",
    );
    put(
        "packages/ones.entpkg/kernels/one.ent",
        "kernel ones.one@1 {\n    requires { field.write; }\n    output result: Rows<Float>;\n    compute { return 1.0; }\n}\n",
    );
    put(
        "play/scene.esn",
        "use \"ones.entpkg\";\n\nnode one: ones.one@1 { domain = 1; }\noutput result <- one.result;\n",
    );
    let store: Arc<dyn Store> = Arc::new(files);
    let loaded = Session::open_loaded(
        store,
        Path::new("play/scene.esn"),
        None,
        "cpu",
        Limits::default(),
        Wait::blocking(),
        false,
    )
    .expect("the scene finds the package by its name");
    assert!(loaded.session.graph().is_some());
}
