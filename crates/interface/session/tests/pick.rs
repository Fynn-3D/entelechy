//! `pick`, the index a condition holds for or the one of least key, and
//! `..zero()`, a struct's every field not written zero: what they give,
//! and that they are the `loop` and the literal they stand for.

use std::path::PathBuf;

fn scratch(name: &str, scene: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-pick-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.osn"), scene).unwrap();
    dir
}

/// The Ints node `n` gives, each of its elements, after one evaluation.
fn ints(name: &str, scene: &str, elements: u32) -> Result<Vec<i32>, String> {
    let dir = scratch(name, scene);
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    if session.graph().is_none() {
        return Err(format!("{events:?}"));
    }
    session
        .evaluate(&orla_kernel::Context::default())
        .map_err(|e| e.to_string())?;
    let out = (0..elements)
        .map(|k| {
            let (_, words) = session
                .element_field(&orla_graph::NodeId("n".into()), "result", k, &[])
                .unwrap();
            words[0] as i32
        })
        .collect();
    let _ = std::fs::remove_dir_all(&dir);
    Ok(out)
}

const DATA: &str = "node data: kernel {
    requires { field.write; }
    output result: Rows<Float>;
    domain = 8;
    compute { return match element() { 0 => 5.0, 1 => 2.0, 2 => 9.0, 3 => 2.0, 4 => -1.0, 5 => 7.0, 6 => 2.0, _ => 3.0 }; }
}
";

#[test]
fn pick_gives_the_first_index_or_the_least_key_or_none() {
    let scene = format!(
        "{DATA}node n: kernel {{
    requires {{ field.read; field.write; }}
    input data: Rows<Float> <- data.result;
    output result: Rows<Int>;
    domain = 6;
    compute {{
        return match element() {{
            0 => pick i < count(data) where at(data, i) > 6.0,
            1 => pick i < count(data) where at(data, i) > 100.0,
            2 => pick i < count(data) where at(data, i) >= 0.0 by at(data, i),
            3 => pick i < count(data) where true by 0 - i,
            4 => pick i < 8 where i > 2 by abs(i - 5),
            _ => pick i < 0 + count(data) where false by i
        }};
    }}
}}
output result <- n.result;
"
    );
    // The first over 6 is at 2; none is over 100; of those not below
    // zero the least is 2.0, first at 1 (and at 3 and 6 after it); the
    // least of -i is at the last; the nearest 5 past 2 is 5; nothing.
    assert_eq!(ints("values", &scene, 6), Ok(vec![2, -1, 1, 7, 5, -1]));
}

/// The first is had as soon as it is found: a loop of a thousand within
/// a budget of five, its third index the one.
#[test]
fn pick_without_a_key_stops_at_the_first() {
    let scene = "node n: kernel {
    requires { field.write; }
    budget 5;
    output result: Rows<Int>;
    domain = 1;
    compute { return pick i < 1000 where i * i >= 4; }
}
output result <- n.result;
";
    assert_eq!(ints("first", scene, 1), Ok(vec![2]));
}

/// Its index goes by its own name; being a loop, `index()` within it is
/// its own, so an enclosing loop's index is named before it. Inside a
/// function too; and a function named `pick` is still called by its name.
#[test]
fn pick_names_its_index_in_loops_and_functions() {
    let scene = "fn pick(x: Int) -> Int { return x * 10; }
fn nearest(to: Int) -> Int { return pick i < 9 where i != to by abs(i * 3 - to); }
node n: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    domain = 3;
    compute {
        let rows = loop 3 with sum = 0 { let k = index(); sum + (pick i < 5 where i > k && index() == i by i) };
        return match element() { 0 => rows, 1 => nearest(7), _ => pick(4) };
    }
}
output result <- n.result;
";
    // Past 0, 1 and 2: 1 + 2 + 3; the multiple of 3 nearest 7 other than
    // at 7 is 6, at 2; `pick(4)` the function.
    assert_eq!(ints("names", scene, 3), Ok(vec![6, 2, 40]));
}

/// `..zero()` writes every field not given zero, a struct and an array
/// within it too; a function of the document named `zero` stands for it.
#[test]
fn zero_writes_the_fields_not_given() {
    let scene = "struct Inner { a: Int; b: IVec2; }
struct Outer { x: Int; inner: Inner; list: [Int; 3]; y: Float; }
node n: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    domain = 4;
    compute {
        let o = Outer { ..zero(), x: 7 };
        return match element() { 0 => o.x, 1 => o.inner.a + o.inner.b.y, 2 => o.list[2], _ => int(o.y) };
    }
}
output result <- n.result;
";
    assert_eq!(ints("zero", scene, 4), Ok(vec![7, 0, 0, 0]));
    let own = "struct Pair { a: Int; b: Int; }
fn zero() -> Pair { return Pair { a: 4, b: 5 }; }
node n: kernel {
    requires { field.write; }
    output result: Rows<Int>;
    domain = 1;
    compute { let p = Pair { ..zero(), a: 1 }; return p.a * 10 + p.b; }
}
output result <- n.result;
";
    assert_eq!(ints("own-zero", own, 1), Ok(vec![15]));
}
