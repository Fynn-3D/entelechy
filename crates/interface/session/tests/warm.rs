//! Sessions one after another over one [`Warm`]: each opens fresh, and
//! what they share (each Kernel prepared once, documents parsed and
//! Kernels compiled once while what they are made from is unchanged)
//! answers as a cold session does. A function changed between them, or a
//! Kernel's text, is compiled again.

use std::path::Path;
use std::sync::Arc;

use orla_kernel::Context;
use orla_session::{Limits, Memory, Session, Store, Warm};

const SCENE: &str =
    "use \"k.orla\";\nnode values: t.values@1 { domain = 4; }\noutput result <- values.result;\n";

fn kernels(scale: &str, offset: &str) -> String {
    format!(
        "fn scaled(x: Float) -> Float {{ return x * {scale}; }}
kernel t.values@1 {{
    requires {{ field.write; }}
    output result: Rows<Float>;
    compute {{ return scaled(float(element())) + {offset}; }}
}}
"
    )
}

fn values(store: &Arc<Memory>, warm: &Warm) -> Vec<f32> {
    let store: Arc<dyn Store> = Arc::clone(store) as Arc<dyn Store>;
    let mut session = Session::open_in(store, Path::new("/w/scene.osn"), None, warm.runtime());
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
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

fn warm(backend: &str) -> Warm {
    Warm::open(
        backend,
        Limits {
            words: 1 << 20,
            work: 1 << 30,
            small: 0,
        },
    )
    .unwrap()
}

#[test]
fn sessions_over_one_warm_answer_alike_and_see_what_changed_between_them() {
    for backend in ["cpu", "jit"] {
        let store = Arc::new(Memory::default());
        store.put(Path::new("/w/scene.osn"), SCENE.as_bytes().to_vec());
        store.put(Path::new("/w/k.orla"), kernels("2.0", "0.5").into_bytes());
        let warm = warm(backend);
        assert_eq!(values(&store, &warm), [0.5, 2.5, 4.5, 6.5], "{backend}");
        assert_eq!(values(&store, &warm), [0.5, 2.5, 4.5, 6.5], "{backend}");
        // The function the Kernel calls, changed: compiled again.
        store.put(Path::new("/w/k.orla"), kernels("3.0", "0.5").into_bytes());
        assert_eq!(values(&store, &warm), [0.5, 3.5, 6.5, 9.5], "{backend}");
        // The Kernel's own text, changed.
        store.put(Path::new("/w/k.orla"), kernels("3.0", "1.0").into_bytes());
        assert_eq!(values(&store, &warm), [1.0, 4.0, 7.0, 10.0], "{backend}");
        // And back: what was compiled and prepared for it before serves.
        store.put(Path::new("/w/k.orla"), kernels("2.0", "0.5").into_bytes());
        assert_eq!(values(&store, &warm), [0.5, 2.5, 4.5, 6.5], "{backend}");
    }
}
