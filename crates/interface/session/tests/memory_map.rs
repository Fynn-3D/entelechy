//! A session's memory map names every Field by what holds it: an input,
//! a node's output, a port's state; each Field once, the largest first,
//! and the Host's census adds up to what they hold and what is loose.

use entelechy_kernel::Context;
use entelechy_session::Session;

const SCENE: &str = "
input word: Rows<Int> = text \"memory\";
kernel t.wide@1 { requires { field.write; } output result: Rows<Int>; compute { return element(); } }
kernel t.count@1 { requires { field.read; field.write; } input before: Rows<Int>; output result: Rows<Int>; compute { return sample(before) + 1; } }
kernel t.both@1 { requires { field.read; field.write; } input a: Rows<Int>; input b: Rows<Int>; output result: Rows<Int>; compute { return sample(a) + at(b, 0); } }
node wide: t.wide@1 { domain = 1000; }
node zero: t.wide@1 { domain = 10; }
node tally: t.count@1 { before <~ tally.result from zero.result; }
node both: t.both@1 { a <- wide.result; b <- word; }
output result <- both.result;
output tally <- tally.result;
";

#[test]
fn the_map_names_every_field_by_what_holds_it() {
    let dir = std::env::temp_dir().join(format!("entelechy-memory-map-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.sync();
    session.evaluate(&Context::default()).unwrap();
    let map = session.memory_map();
    let bytes_of = |owner: &str| {
        map.held
            .iter()
            .find(|entry| entry.owner == owner)
            .unwrap_or_else(|| panic!("{owner} is held: {:#?}", map.held))
    };
    assert_eq!(bytes_of("word").kind, "input");
    assert_eq!(
        bytes_of("word").bytes.host,
        6 * 4,
        "six characters, a word each"
    );
    // `wide` runs fused into `both`, which holds the thousand words.
    assert_eq!(bytes_of("both.result").bytes.host, 1000 * 4);
    assert_eq!(
        bytes_of("tally.result").kind,
        "state",
        "state claims it before the output"
    );
    assert_eq!(
        map.held[0].owner, "both.result",
        "the largest first: {:#?}",
        map.held
    );
    let owners: Vec<&str> = map.held.iter().map(|entry| entry.owner.as_str()).collect();
    let mut unique = owners.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), owners.len(), "each Field once: {owners:?}");
    let held: u64 = map.held.iter().map(|entry| entry.bytes.host).sum();
    assert_eq!(
        held + map.loose.host,
        map.census.host + map.census.kept,
        "what is held and what is loose make up the Host's count"
    );
    assert_eq!(
        map.census.device, 0,
        "the processor keeps no Field on a device"
    );
}
