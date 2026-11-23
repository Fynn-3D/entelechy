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
    // The first turn queues the scene's five Kernels and prepares them
    // for a moment.
    session.sync_a_turn();
    assert!(session.scene_waiting(), "the scene waits for its Kernels");
    assert!(session.preparing_left() > 0);
    let mut syncs = 1;
    while session.scene_waiting() {
        session.sync_a_turn();
        syncs += 1;
        assert!(syncs < 20, "the turns never finish");
    }
    // More than one sync: never inside the sync that found them.
    assert!(syncs >= 2, "{syncs} syncs");
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
    assert!(syncs >= 2, "{syncs} syncs");
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

#[test]
fn finishing_used_kernels_warms_a_bounded_window_and_skips_unreached_kernels() {
    use std::collections::BTreeSet;
    use std::sync::{Mutex, OnceLock};
    use std::time::Duration;

    use entelechy_abi::{Handle, Invocation, Scope};
    use entelechy_ir::Validated;
    use entelechy_kernel::{Backend, ExecutionError, Prepared, Runtime};

    #[derive(Default)]
    struct WindowedDevice {
        in_flight: Mutex<BTreeSet<String>>,
        max_in_flight: Mutex<usize>,
        warmed: Mutex<Vec<String>>,
        finished: Mutex<Vec<String>>,
    }

    struct Slot {
        inner: Box<dyn Prepared>,
        id: String,
        warmed: OnceLock<()>,
        built: OnceLock<()>,
    }

    impl Prepared for Slot {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

    impl Backend for &'static WindowedDevice {
        fn name(&self) -> &'static str {
            "windowed"
        }
        fn has_device(&self) -> bool {
            true
        }
        fn prepare(&self, module: &Validated) -> Result<Box<dyn Prepared>, String> {
            Ok(Box::new(Slot {
                inner: entelechy_cpu::CpuBackend.prepare(module)?,
                id: module.manifest.id.to_string(),
                warmed: OnceLock::new(),
                built: OnceLock::new(),
            }))
        }
        fn warm(&self, prepared: &dyn Prepared) {
            let slot = prepared.as_any().downcast_ref::<Slot>().unwrap();
            if slot.built.get().is_some() {
                return;
            }
            slot.warmed.get_or_init(|| {
                self.warmed.lock().unwrap().push(slot.id.clone());
                let mut active = self.in_flight.lock().unwrap();
                active.insert(slot.id.clone());
                let mut peak = self.max_in_flight.lock().unwrap();
                *peak = (*peak).max(active.len());
            });
        }
        fn finish(&self, prepared: &dyn Prepared) {
            let slot = prepared.as_any().downcast_ref::<Slot>().unwrap();
            slot.built.get_or_init(|| {
                std::thread::sleep(Duration::from_millis(14));
                self.in_flight.lock().unwrap().remove(&slot.id);
                self.finished.lock().unwrap().push(slot.id.clone());
            });
        }
        fn finished(&self, prepared: &dyn Prepared) -> bool {
            prepared
                .as_any()
                .downcast_ref::<Slot>()
                .is_none_or(|slot| slot.built.get().is_some())
        }
        fn execute(
            &self,
            prepared: &dyn Prepared,
            module: &Validated,
            invocation: &Invocation,
            scope: &mut Scope<'_>,
        ) -> Result<Vec<Handle>, ExecutionError> {
            let inner = &prepared.as_any().downcast_ref::<Slot>().unwrap().inner;
            Backend::execute(
                &entelechy_cpu::CpuBackend,
                inner.as_ref(),
                module,
                invocation,
                scope,
            )
        }
    }

    let dir = std::env::temp_dir().join(format!("entelechy-turns-window-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let mut text = String::from(
        "kernel t.unused@1 { requires { field.write; } output result: Rows<Int>; compute { return 99; } }\n\
         kernel t.k0@1 { requires { field.write; } output result: Rows<Int>; compute { return element(); } }\n",
    );
    for i in 1..18 {
        text.push_str(&format!(
            "kernel t.k{i}@1 {{ requires {{ field.read; field.write; }} input x: Rows<Int>; output result: Rows<Int>; compute {{ return sample(x) + 1; }} }}\n"
        ));
    }
    text.push_str("node n0: t.k0@1 { domain = 64; }\n");
    for i in 1..18 {
        let prev = i - 1;
        text.push_str(&format!(
            "node n{i}: t.k{i}@1 {{ x <- n{prev}.result; domain = 64; }}\n"
        ));
    }
    text.push_str("output result <- n17.result;\n");
    std::fs::write(dir.join("scene.esn"), text).unwrap();

    let device: &'static WindowedDevice = Box::leak(Box::default());
    let mut runtime = Runtime::with_backends(
        1 << 20,
        Box::new(device),
        Some(Box::new(entelechy_cpu::CpuBackend)),
    );
    runtime.set_small(0);
    let mut session = Session::open_with(&dir.join("scene.esn"), None, runtime);
    session.prepare_in_turns();

    let mut turns = 0;
    while {
        session.sync_a_turn();
        turns += 1;
        session.scene_waiting()
    } {
        assert!(turns < 40, "finishing turns never completed");
    }
    // 18 reachable Kernels at ~14ms per finish across 12ms turns must slice across multiple turns.
    assert!(
        turns >= 18,
        "expected turn-sliced finishing, ran in {turns} turns"
    );
    let peak = *device.max_in_flight.lock().unwrap();
    assert!(
        (1..=12).contains(&peak),
        "expected sliding warm window <= 12, saw peak {peak}"
    );
    let warmed = device.warmed.lock().unwrap().clone();
    let finished = device.finished.lock().unwrap().clone();
    assert!(!warmed.iter().any(|id| id == "t.unused@1"), "{warmed:?}");
    assert!(
        !finished.iter().any(|id| id == "t.unused@1"),
        "{finished:?}"
    );
    assert_eq!(finished.len(), 18, "{finished:?}");
    assert_eq!(result(&mut session)[0], 17);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sync_turn_checked_surfaces_bundle_landing_failure() {
    let dir = scene("bundle_fail");
    let mut at_once = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    at_once.sync();
    let mut bundle = at_once
        .bundle_with_fonts(|_, _, _| unreachable!("no font nodes"))
        .unwrap();
    // Duplicate the last module so `open_bundle(..., turns = true)` queues
    // tickets across turns and `land_pending_bundle` fails on `publish_all`.
    let dup = bundle.modules.last().unwrap().clone();
    bundle.modules.push(dup);

    let err = match Session::open_bundle(
        bundle,
        "cpu",
        entelechy_session::Limits::default(),
        entelechy_session::Wait::blocking(),
        true,
    ) {
        Err(why) => why,
        Ok(mut loaded) => {
            let mut outcome = None;
            for _ in 0..20 {
                if let Err(why) = loaded.session.sync_turn_checked() {
                    // Subsequent turns keep returning the failure instead of reporting 0 waiting.
                    assert_eq!(loaded.session.sync_turn_checked().unwrap_err(), why);
                    outcome = Some(why);
                    break;
                }
            }
            outcome.expect("bundle landing failure must surface")
        }
    };
    assert!(err.contains("t.") || err.contains("already"), "{err}");

    // And a bundle missing a referenced Kernel is rejected at landing time too.
    let mut missing = at_once
        .bundle_with_fonts(|_, _, _| unreachable!("no font nodes"))
        .unwrap();
    missing.modules.pop();
    let missing_err = match Session::open_bundle(
        missing,
        "cpu",
        entelechy_session::Limits::default(),
        entelechy_session::Wait::blocking(),
        true,
    ) {
        Err(why) => why,
        Ok(mut loaded) => {
            let mut outcome = None;
            for _ in 0..20 {
                if let Err(why) = loaded.session.sync_turn_checked() {
                    outcome = Some(why);
                    break;
                }
            }
            outcome.expect("missing Kernel in bundle must fail landing")
        }
    };
    assert!(missing_err.contains("not published"), "{missing_err}");
    let _ = std::fs::remove_dir_all(&dir);
}
