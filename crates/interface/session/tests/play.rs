//! The Player's frame with no window (`orla_session::play`): what every
//! interface that plays a scene shares. The keys' meanings, the preview,
//! the line of frame data, the pointer found in the picture, which
//! output is the sound, a failure said once, and a replay played to its
//! end, each held here without a device.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use orla_session::Session;
use orla_session::play::{Asked, Frame, Placement, Play, View};

const KERNELS: &str = "
kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }
kernel t.count@1 { requires { field.read; field.write; } input last: Rows<Int>; output result: Rows<Int>; compute { return sample(last) + 1; } }
kernel t.picture@1 { requires { field.read; field.write; input.read; } input count: Rows<Int>; output result: Image; compute { let n = float(at(count, 0)); let a = if key(65) { 1.0 } else { 0.0 }; return vec4(n / 100.0, a, 0.0, 1.0); } }
kernel t.tone@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.25; } }
";

const SCENE: &str = "
use \"k.orla\";
node zero: t.zero@1 { domain = 1; }
node count: t.count@1 { last <~ count.result from zero.result; domain = 1; }
node picture: t.picture@1 { count <- count.result; domain = vec2(4, 2); }
node tone: t.tone@1 { domain = 16; }
output view <- picture.result;
output sound <- tone.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-play-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.orla"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.osn"), SCENE).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.osn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// A clock that moves a second each time it is read, so a replay's
/// times come at once.
fn racing() -> f64 {
    static TICKS: AtomicU64 = AtomicU64::new(0);
    TICKS.fetch_add(1, Ordering::Relaxed) as f64
}

fn still() -> f64 {
    0.0
}

const VIEW: View = View {
    window: [8.0, 4.0],
    scale: 1.0,
    rate: 0.0,
};

fn shown(play: &mut Play, session: &mut Session) -> Box<orla_session::play::Shown> {
    match play.frame(session, VIEW) {
        Frame::Shown(shown) => shown,
        Frame::Wait => panic!("a frame waited"),
        Frame::Failed => panic!("a frame failed: {}", play.status()),
    }
}

/// The red of the picture's first pixel: the count over a hundred.
fn red(session: &mut Session, picture: orla_kernel::Handle) -> f32 {
    session.field(picture).unwrap();
    f32::from_bits(session.runtime().host().element(picture, 0).unwrap()[0])
}

/// The picture is the first grid of pixels and the sound the first rows
/// of floats; F5 starts the state again; a letter reaches the scene's
/// `key`; Tab shows the nodes one by one and then the picture again;
/// F10 shows the line of frame data; Escape asks the device to stop.
#[test]
fn the_keys_mean_what_they_mean_in_every_interface() {
    let dir = scratch("keys");
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let first = shown(&mut play, &mut session);
    assert!(first.sound.is_some(), "rows of floats are the sound");
    assert!((red(&mut session, first.picture) - 0.01).abs() < 1e-6);
    shown(&mut play, &mut session);
    let third = shown(&mut play, &mut session);
    assert!((red(&mut session, third.picture) - 0.03).abs() < 1e-6);
    assert_eq!(play.key(&mut session, "F5", true), None);
    assert!(play.said().iter().any(|line| line == "state reset"));
    let again = shown(&mut play, &mut session);
    assert!(
        (red(&mut session, again.picture) - 0.01).abs() < 1e-6,
        "F5 starts over"
    );

    play.key(&mut session, "a", true);
    let pressed = shown(&mut play, &mut session);
    session.field(pressed.picture).unwrap();
    let green = f32::from_bits(
        session
            .runtime()
            .host()
            .element(pressed.picture, 0)
            .unwrap()[1],
    );
    assert_eq!(green, 1.0, "the letter reached the scene");
    play.key(&mut session, "a", false);

    play.key(&mut session, "Tab", true);
    assert!(play.said().iter().any(|line| line == "previewing zero"));
    let previewed = shown(&mut play, &mut session);
    let zero = session.cached(&orla_graph::NodeId("zero".into()), "result");
    assert_eq!(Some(previewed.picture), zero, "Tab shows the first node");
    for _ in 0..4 {
        play.key(&mut session, "Tab", true);
    }
    assert!(play.said().iter().any(|line| line == "previewing output"));
    let back = shown(&mut play, &mut session);
    assert_eq!(
        back.picture, back.report.outputs[0].1,
        "and then the picture"
    );

    assert!(play.panel().is_none());
    play.key(&mut session, "F10", true);
    shown(&mut play, &mut session);
    let panel = play.panel().expect("F10 shows the line");
    assert!(panel.width > 0 && panel.height > 0);
    assert_eq!(play.key(&mut session, "Escape", true), Some(Asked::Stop));
}

/// The pointer is found in the picture through where the device said it
/// landed, clamped to its edges.
#[test]
fn the_pointer_is_found_in_the_picture_where_it_landed() {
    let dir = scratch("pointer");
    let mut session = open(&dir);
    let mut play = Play::new(still);
    shown(&mut play, &mut session);
    let landed = Placement {
        x: 10,
        y: 0,
        width: 100,
        height: 50,
    };
    play.placed(Some(landed), (4, 2), "presented on the device", false, 0.0);
    play.pointer(60.0, 25.0);
    assert_eq!(play.context(VIEW).input.pointer, [2.0, 1.0]);
    play.pointer(0.0, 0.0);
    assert_eq!(
        play.context(VIEW).input.pointer,
        [0.0, 0.0],
        "the letterbox clamps"
    );
    // Where it was, when the device cannot say this frame.
    play.placed(None, (4, 2), "window not ready", false, 0.0);
    play.pointer(60.0, 25.0);
    assert_eq!(play.context(VIEW).input.pointer, [2.0, 1.0]);
    play.button(1, true);
    play.wheel(0.0, 2.0);
    let context = play.context(VIEW);
    assert!(context.input.pointer_right);
    assert_eq!(context.input.wheel, [0.0, 2.0]);
}

/// A failure every frame is said once, and a frame that works again says
/// so.
#[test]
fn a_failure_is_said_once() {
    let dir = scratch("failure");
    let mut session = open(&dir);
    let mut play = Play::new(still);
    shown(&mut play, &mut session);
    let broken = SCENE.replace("domain = vec2(4, 2);", "domain = window() / 0.0;");
    std::fs::write(dir.join("scene.osn"), broken).unwrap();
    let mut failures = 0;
    for _ in 0..3 {
        if matches!(play.frame(&mut session, VIEW), Frame::Failed) {
            failures += 1;
        }
    }
    assert_eq!(failures, 3);
    let said: Vec<String> = play.said();
    let failed: Vec<&String> = said
        .iter()
        .filter(|line| line.contains("evaluation failed"))
        .collect();
    assert_eq!(failed.len(), 1, "{said:?}");
    std::fs::write(dir.join("scene.osn"), SCENE).unwrap();
    shown(&mut play, &mut session);
    assert!(play.said().iter().any(|line| line == "evaluating again"));
}

/// A replay drives the scene at its own times, its checks run as they
/// come, and at its end what it observed is said and the person has the
/// scene.
#[test]
fn a_replay_plays_to_its_end_and_says_what_it_saw() {
    let replay_path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/ant/tests/ant.replay");
    let text = std::fs::read_to_string(&replay_path).unwrap();
    let replay = orla_session::replay::parse(&text, replay_path.parent().unwrap()).unwrap();
    let mut session = Session::open(&replay.scene, replay.kernels.as_deref(), 1 << 22);
    session.sync();
    let mut play = Play::new(racing);
    play.replay(&replay, &mut session);
    let mut said = play.said();
    for _ in 0..100 {
        if !play.replaying() {
            break;
        }
        let _ = play.frame(&mut session, VIEW);
        said.extend(play.said());
    }
    assert!(!play.replaying(), "the replay ended");
    assert!(
        said.iter()
            .any(|line| line.starts_with("replay: ok (2 frames, 14 checks)")),
        "{said:?}"
    );
}

/// A press shorter than a frame is seen by the next frame, and released
/// after it: a quick hand is not lost between two frames.
#[test]
fn a_press_lasts_a_frame_however_quick() {
    let dir = scratch("quick");
    let mut session = open(&dir);
    let mut play = Play::new(still);
    shown(&mut play, &mut session);
    play.button(0, true);
    play.button(0, false);
    play.key(&mut session, "a", true);
    play.key(&mut session, "a", false);
    let pressed = shown(&mut play, &mut session);
    session.field(pressed.picture).unwrap();
    let green = f32::from_bits(
        session
            .runtime()
            .host()
            .element(pressed.picture, 0)
            .unwrap()[1],
    );
    assert_eq!(green, 1.0, "the frame after a quick tap saw the key");
    let after = play.context(VIEW).input;
    assert!(
        !after.pointer_down && !after.key(65),
        "and the next one sees it let go"
    );
    // A press held across frames is released when it is let go.
    play.button(0, true);
    shown(&mut play, &mut session);
    play.button(0, false);
    assert!(!play.context(VIEW).input.pointer_down);
}

/// A placed graph's node carries its place in its id (`fx.gray`), so a
/// replay naming its port means the last dot: every step that names a
/// port reads it the same way.
#[test]
fn a_placed_nodes_port_is_what_follows_the_last_dot() {
    use orla_session::replay::{Step, parse};
    let text = "scene scene.osn\nframe 0.0\nreset fx.gray.result\nexpect count fx.gray.result = 4\nexpect fx.gray.result[0].size = 1\n";
    let replay = parse(text, Path::new(".")).unwrap();
    let named: Vec<(String, String)> = replay
        .steps
        .iter()
        .filter_map(|step| match step {
            Step::ResetPort { node, port } | Step::ExpectCount { node, port, .. } => {
                Some((node.clone(), port.clone()))
            }
            Step::ExpectElement { node, port, .. } => Some((node.clone(), port.clone())),
            _ => None,
        })
        .collect();
    let placed = ("fx.gray".to_owned(), "result".to_owned());
    assert_eq!(named, vec![placed.clone(), placed.clone(), placed]);
    // A name with no port is refused where a port is needed.
    assert!(parse("scene scene.osn\nreset gray\n", Path::new(".")).is_err());
}
