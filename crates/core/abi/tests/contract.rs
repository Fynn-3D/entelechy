//! The capabilities a Kernel may declare are the ABI: a source written
//! against a version names these, and a Host grants no other. Adding,
//! dropping or renaming one is an ABI change, made here together with
//! `ABI_VERSION`, never by accident.

use orla_abi::{ABI_VERSION, Capability};

#[test]
fn the_capabilities_are_those_of_the_abi_version() {
    let names: Vec<&str> = Capability::ALL.iter().map(|c| c.name()).collect();
    assert_eq!(
        (ABI_VERSION, names),
        (
            1,
            vec!["field.read", "field.write", "time.read", "input.read"]
        ),
        "the capabilities changed; change ABI_VERSION with them and this pin"
    );
    for capability in Capability::ALL {
        assert_eq!(capability.name().parse::<Capability>(), Ok(capability));
    }
}
