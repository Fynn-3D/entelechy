//! The Player's frame without a window (`entelechy_session::play`), shared by
//! every interface that plays a scene: keys, preview, frame-data line,
//! pointer placement, the sound output, failures reported once, and
//! replays played to their end.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use entelechy_session::Session;
use entelechy_session::play::{Frame, Placement, Play, View};

const KERNELS: &str = "
kernel t.zero@1 { requires { field.write; } output result: Rows<Int>; compute { return 0; } }
kernel t.count@1 { requires { field.read; field.write; } input last: Rows<Int>; output result: Rows<Int>; compute { return sample(last) + 1; } }
kernel t.picture@1 { requires { field.read; field.write; input.read; } input count: Rows<Int>; output result: Image; compute { let n = float(at(count, 0)); let a = if key(65) { 1.0 } else { 0.0 }; return vec4(n / 100.0, a, 0.0, 1.0); } }
kernel t.tone@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.25; } }
";

const SCENE: &str = "
use \"k.ent\";
node zero: t.zero@1 { domain = 1; }
node count: t.count@1 { last <~ count.result from zero.result; domain = 1; }
node picture: t.picture@1 { count <- count.result; domain = vec2(4, 2); }
node tone: t.tone@1 { domain = 16; }
output view <- picture.result;
output sound <- tone.result;
";

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-play-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.ent"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

/// A clock that advances a second per read, so a replay's times come at
/// once.
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
    queued: 0,
};

fn shown(play: &mut Play, session: &mut Session) -> Box<entelechy_session::play::Shown> {
    match play.frame(session, VIEW) {
        Frame::Shown(shown) => shown,
        Frame::Wait => panic!("a frame waited"),
        Frame::Failed => panic!("a frame failed: {}", play.status()),
    }
}

/// The red of the picture's first pixel: the count over a hundred.
fn red(session: &mut Session, picture: entelechy_kernel::Handle) -> f32 {
    session.field(picture).unwrap();
    f32::from_bits(session.runtime().host().element(picture, 0).unwrap()[0])
}

/// The picture is the first pixel grid and the sound the first float
/// rows; F5 restarts the state; a letter, Tab and Escape reach the
/// scene's `key`; F6 cycles the nodes and back to the picture; F10 shows
/// the frame-data line, naming the port shown.
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
    play.key(&mut session, "F5", true);
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

    play.key(&mut session, "F6", true);
    assert!(play.said().iter().any(|line| line == "previewing zero"));
    let previewed = shown(&mut play, &mut session);
    let zero = session.cached(&entelechy_graph::NodeId("zero".into()), "result");
    assert_eq!(Some(previewed.picture), zero, "F6 shows the first node");
    assert_eq!(
        play.showing(&previewed.report, previewed.picture),
        "view zero.result"
    );
    for _ in 0..4 {
        play.key(&mut session, "F6", true);
    }
    assert!(play.said().iter().any(|line| line == "previewing output"));
    let back = shown(&mut play, &mut session);
    assert_eq!(
        back.picture, back.report.outputs[0].1,
        "and then the picture"
    );
    assert_eq!(play.showing(&back.report, back.picture), "output view");

    assert!(play.panel().is_none());
    play.key(&mut session, "F10", true);
    shown(&mut play, &mut session);
    let panel = play.panel().expect("F10 shows the line");
    assert!(panel.width > 0 && panel.height > 0);
    // Tab and Escape are the scene's: neither previews nor stops.
    play.key(&mut session, "Tab", true);
    play.key(&mut session, "Escape", true);
    let held = play.context(VIEW).input;
    assert!(
        held.key(9) && held.key(27),
        "Tab and Escape reach the scene"
    );
}

/// The pointer is placed in the picture from where the device says it
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
    // The last place, when the device cannot say this frame.
    play.placed(None, (4, 2), "window not ready", false, 0.0);
    play.pointer(60.0, 25.0);
    assert_eq!(play.context(VIEW).input.pointer, [2.0, 1.0]);
    play.button(1, true);
    play.wheel(0.0, 2.0);
    let context = play.context(VIEW);
    assert!(context.input.pointer_right);
    assert_eq!(context.input.wheel, [0.0, 2.0]);
}

/// A clock a second on at each look: every frame looks at the disk.
fn ticking() -> f64 {
    static SECONDS: AtomicU64 = AtomicU64::new(0);
    SECONDS.fetch_add(1, Ordering::Relaxed) as f64
}

/// A failure every frame is reported once, and recovery is reported.
#[test]
fn a_failure_is_said_once() {
    let dir = scratch("failure");
    let mut session = open(&dir);
    let mut play = Play::new(ticking);
    shown(&mut play, &mut session);
    let broken = SCENE.replace("domain = vec2(4, 2);", "domain = window() / 0.0;");
    std::fs::write(dir.join("scene.esn"), broken).unwrap();
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
    std::fs::write(dir.join("scene.esn"), SCENE).unwrap();
    shown(&mut play, &mut session);
    assert!(play.said().iter().any(|line| line == "evaluating again"));
}

/// A replay runs its checks as they come; at its end its observations are
/// reported and the person has the scene.
#[test]
fn a_replay_plays_to_its_end_and_says_what_it_saw() {
    let dir = scratch("replay");
    // The count goes on from frame to frame; the picture's red is its
    // hundredth.
    let text = "scene scene.esn
frame 0.0
expect count.result[0] = 1
expect 0 0 = 0.01 0 0 0.001
frame 0.05
expect count.result[0] = 2
expect 0 0 = 0.02 0 0 0.001
";
    let replay = entelechy_session::replay::parse(text, &dir).unwrap();
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
            .any(|line| line.starts_with("replay: ok (2 frames, 4 checks)")),
        "{said:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A press shorter than a frame is seen by the next frame and released
/// after it.
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
    // A press held across frames is released when let go.
    play.button(0, true);
    shown(&mut play, &mut session);
    play.button(0, false);
    assert!(!play.context(VIEW).input.pointer_down);
}

#[test]
fn keys_whose_release_went_elsewhere_are_let_go() {
    let dir = scratch("released");
    let mut session = open(&dir);
    let mut play = Play::new(still);
    shown(&mut play, &mut session);
    // Shift and Command pressed here, then the focus gone with their
    // releases: the window let go of every key.
    play.key(&mut session, "Shift", true);
    play.key(&mut session, "Super", true);
    play.key(&mut session, "a", true);
    shown(&mut play, &mut session);
    play.release_keys();
    let input = play.context(VIEW).input;
    assert!(!input.key(16) && !input.key(91) && !input.key(65));
    // The system saying Command is no longer held lets it go too.
    play.key(&mut session, "Super", true);
    play.key(&mut session, "Shift", true);
    shown(&mut play, &mut session);
    play.modifiers(true, false, false, false);
    let input = play.context(VIEW).input;
    assert!(input.key(16) && !input.key(91));
}

/// A placed graph's node id carries its place (`fx.gray`), so every
/// replay step naming a port takes it after the last dot.
#[test]
fn a_placed_nodes_port_is_what_follows_the_last_dot() {
    use entelechy_session::replay::{Step, parse};
    let text = "scene scene.esn\nframe 0.0\nreset fx.gray.result\nexpect count fx.gray.result = 4\nexpect fx.gray.result[0].size = 1\n";
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
    assert!(parse("scene scene.esn\nreset gray\n", Path::new(".")).is_err());
}

/// The text a device types (a key's, or an input method's commit) reaches
/// `typed()` on the next frame and that frame alone; control characters
/// and the private codes some systems give arrows are keys, not text.
#[test]
fn text_typed_reaches_the_next_frame_alone() {
    let dir = scratch("typed");
    std::fs::write(
        dir.join("k.ent"),
        "kernel t.chars@1 { requires { field.write; input.read; } output result: Rows<Int>; compute { return typed(element()); } }
kernel t.blank@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }
",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.esn"),
        "use \"k.ent\";\nnode blank: t.blank@1 { domain = vec2(1, 1); }\nnode chars: t.chars@1 { domain = 3; }\noutput view <- blank.result;\noutput chars <- chars.result;\n",
    )
    .unwrap();
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let read = |session: &mut Session| -> Vec<u32> {
        let handle = session.output(Some("chars")).unwrap();
        session.field(handle).unwrap().values().to_vec()
    };
    play.typed("a\r\t\u{8}\u{F700}");
    play.typed("你");
    shown(&mut play, &mut session);
    assert_eq!(read(&mut session), [97, 20320, 0]);
    shown(&mut play, &mut session);
    assert_eq!(read(&mut session), [0, 0, 0]);
    let _ = std::fs::remove_dir_all(&dir);
}
