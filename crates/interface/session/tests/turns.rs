//! Where no thread can start (a page), a session prepares its Kernels a
//! few each sync: the frames wait until the scene's are all prepared, then
//! run as a session that prepared them at once does.

use entelechy_kernel::Context;
use entelechy_session::Session;

const SCENE: &str = "
kernel t.a@1 { requires { field.write; } output result: Rows<Int>; compute { return element() * 2; } }
kernel t.b@1 { requires { field.read; field.write; } input x: Rows<Int>; output result: Rows<Int>; compute { return sample(x) + 1; } }
kernel t.c@1 { requires { field.read; field.write; } input x: Rows<Int>; output result: Rows<Int>; compute { return sample(x) * 3; } }
kernel t.d@1 { requires { field.read; field.write; } input x: Rows<Int>; output result: Rows<Int>; compute { return sample(x) - 4; } }
kernel t.e@1 { requires { field.read; field.write; } input x: Rows<Int>; output result: Rows<Int>; compute { return sample(x) + element(); } }
node a: t.a@1 { domain = 6; }
node b: t.b@1 { x <- a.result; domain = 6; }
node c: t.c@1 { x <- b.result; domain = 6; }
node d: t.d@1 { x <- c.result; domain = 6; }
node e: t.e@1 { x <- d.result; domain = 6; }
output result <- e.result;
";

fn scene(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-turns-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn result(session: &mut Session) -> Vec<u32> {
    session.evaluate(&Context::default()).unwrap();
    let handle = session.output(Some("result")).unwrap();
    session.field(handle).unwrap().values().to_vec()
}

#[test]
fn kernels_prepared_in_turns_wait_then_run_as_at_once() {
    let dir = scene("waits");
    let mut at_once = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    at_once.sync();
    let expected = result(&mut at_once);

    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.prepare_in_turns();
    // The first turn queues the scene's five Kernels and prepares a couple.
    session.sync_a_turn();
    assert!(session.scene_waiting(), "the scene waits for its Kernels");
    assert!(session.preparing_left() > 0);
    let mut syncs = 1;
    while session.scene_waiting() {
        session.sync_a_turn();
        syncs += 1;
        assert!(syncs < 20, "the turns never finish");
    }
    // More than one sync: never all at once.
    assert!(syncs > 2, "{syncs} syncs");
    assert_eq!(session.preparing_left(), 0);
    assert_eq!(result(&mut session), expected);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_replacement_prepared_in_turns_lands_on_a_later_sync() {
    let dir = scene("replace");
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.prepare_in_turns();
    while {
        session.sync_a_turn();
        session.scene_waiting()
    } {}
    let before = result(&mut session);
    std::fs::write(
        dir.join("scene.esn"),
        SCENE.replace("return sample(x) - 4;", "return sample(x) - 5;"),
    )
    .unwrap();
    // The running version runs until the replacement is prepared.
    let mut syncs = 0;
    loop {
        session.sync_a_turn();
        syncs += 1;
        let now = result(&mut session);
        if now != before {
            assert_eq!(
                now,
                before.iter().map(|v| v.wrapping_sub(1)).collect::<Vec<_>>()
            );
            break;
        }
        assert!(syncs < 20, "the replacement never lands");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn kernels_of_used_documents_are_waited_for_too() {
    let dir = std::env::temp_dir().join(format!("entelechy-turns-used-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    // The Kernels in a directory the scene uses, as a project keeps them.
    let (kernels, nodes) = SCENE.split_at(SCENE.find("node a:").unwrap());
    std::fs::write(dir.join("kernels/all.ent"), kernels).unwrap();
    std::fs::write(dir.join("scene.esn"), format!("use \"kernels/\";\n{nodes}")).unwrap();
    let mut at_once = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    at_once.sync();
    let expected = result(&mut at_once);

    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.prepare_in_turns();
    session.sync_a_turn();
    assert!(
        session.scene_waiting(),
        "the scene waits for the Kernels it uses"
    );
    let mut syncs = 1;
    while session.scene_waiting() {
        session.sync_a_turn();
        syncs += 1;
        assert!(syncs < 20, "the turns never finish");
    }
    assert!(syncs > 2, "{syncs} syncs");
    assert_eq!(result(&mut session), expected);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn held_reads_settle_when_a_first_load_in_turns_lands() {
    // The Kernels used from a package beside the project, reached by path,
    // as a page's copies reach them: read while the first load prepares in
    // turns, and still readable after it lands.
    let dir = std::env::temp_dir().join(format!("entelechy-turns-held-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("game/kernels")).unwrap();
    std::fs::create_dir_all(dir.join("packages/more.entpkg/kernels")).unwrap();
    std::fs::write(
        dir.join("packages/more.entpkg/package.esn"),
        "package more@1 { language = 1; ir = 1; }\nuse \"kernels/\";\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("packages/more.entpkg/kernels/twice.ent"),
        "fn twice(x: Int) -> Int { return x * 2; }\n",
    )
    .unwrap();
    let (kernels, nodes) = SCENE.split_at(SCENE.find("node a:").unwrap());
    std::fs::write(
        dir.join("game/kernels/all.ent"),
        format!(
            "use \"../../packages/more.entpkg\";\n{}",
            kernels.replace("return element() * 2;", "return twice(element());")
        ),
    )
    .unwrap();
    std::fs::write(
        dir.join("game/scene.esn"),
        format!("use \"kernels/\";\n{nodes}"),
    )
    .unwrap();
    let mut session = Session::open(&dir.join("game/scene.esn"), None, 1 << 20);
    session.restrict_reads().unwrap();
    session.prepare_in_turns();
    let mut syncs = 0;
    loop {
        let events = session.sync_a_turn();
        let said: Vec<String> = events.iter().map(ToString::to_string).collect();
        assert!(
            !said.iter().any(|line| line.contains("rejected")),
            "{said:?}"
        );
        syncs += 1;
        if !session.scene_waiting() {
            break;
        }
        assert!(syncs < 20, "the turns never finish");
    }
    assert!(session.graph().is_some());
    assert_eq!(result(&mut session).len(), 6);
    // Settled at the landing: a read elsewhere is held.
    assert!(
        session
            .allowed_read(&std::env::temp_dir().join("elsewhere.txt"))
            .is_err()
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_asked_for_sync_lands_what_it_read_at_once() {
    // Turns are for loading frame by frame; a sync asked for (the protocol's
    // `sync`) answers with what it read in place, as it always has.
    let dir = scene("asked");
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    session.prepare_in_turns();
    session.sync();
    assert!(!session.scene_waiting());
    assert_eq!(session.preparing_left(), 0);
    assert_eq!(result(&mut session).len(), 6);
    let _ = std::fs::remove_dir_all(&dir);
}
