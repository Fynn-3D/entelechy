//! The pointer a scene draws (`entelechy_session::play::PointerLook`): the
//! picture output named `cursor` is the system's pointer in every
//! interface, pointing with the pixel `cursor_hotspot` names, else its
//! middle; an interface is told when it changes and only then, one too
//! large is said once and leaves the system's pointer, and a scene
//! without one keeps the system's.

use std::path::{Path, PathBuf};

use entelechy_session::Session;
use entelechy_session::play::{Frame, Play, PointerLook, View};

const KERNELS: &str = "
kernel t.view@1 { requires { field.write; } output result: Image; compute { return vec4(0.0, 0.0, 0.0, 1.0); } }
kernel t.mark@1 { requires { field.write; input.read; } output result: Image; compute { let r = if key(65) { 0.25 } else { 1.0 }; return vec4(r, 0.5, 0.0, 1.0); } }
kernel t.point@1 { requires { field.write; } param at: Vec2 = vec2(0.0, 0.0); output result: Rows<Vec2>; compute { return at; } }
";

fn scene(cursor: Option<&str>) -> String {
    let mut scene = "use \"k.ent\";\nnode view: t.view@1 { domain = vec2(4, 4); }\n".to_owned();
    if let Some(domain) = cursor {
        scene.push_str(&format!(
            "node mark: t.mark@1 {{ domain = {domain}; }}\noutput view <- view.result;\noutput cursor <- mark.result;\n"
        ));
    } else {
        scene.push_str("output view <- view.result;\n");
    }
    scene
}

fn scratch(name: &str, cursor: Option<&str>) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("entelechy-pointer-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("k.ent"), KERNELS).unwrap();
    std::fs::write(dir.join("scene.esn"), scene(cursor)).unwrap();
    dir
}

fn open(dir: &Path) -> Session {
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let events = session.sync();
    assert!(session.graph().is_some(), "{events:?}");
    session
}

fn still() -> f64 {
    0.0
}

const VIEW: View = View {
    window: [4.0, 4.0],
    scale: 1.0,
    rate: 0.0,
    queued: 0,
};

/// What the pointer's look was told as, frame by frame, for `frames`
/// frames.
fn looks(play: &mut Play, session: &mut Session, frames: usize) -> Vec<Option<PointerLook>> {
    (0..frames)
        .map(|_| match play.frame(session, VIEW) {
            Frame::Shown(shown) => play.pointer_look(session, &shown.report),
            Frame::Wait => panic!("a frame waited"),
            Frame::Failed => panic!("a frame failed: {}", play.status()),
        })
        .collect()
}

fn picture(red: u8) -> PointerLook {
    pointing(red, (1, 1))
}

fn pointing(red: u8, hotspot: (u32, u32)) -> PointerLook {
    PointerLook::Picture {
        width: 3,
        height: 2,
        rgba: [red, 128, 0, 255].repeat(6),
        hotspot,
    }
}

/// The scene's picture is the pointer, told once until it changes; a
/// change the scene makes is told again.
#[test]
fn the_cursor_output_is_the_pointer_and_told_when_it_changes() {
    let dir = scratch("told", Some("vec2(3, 2)"));
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 4)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![picture(255)]);

    play.key(&mut session, "A", true);
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 4)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![picture(64)]);
}

/// A scene without the output keeps the system's pointer, and one that
/// takes it away gives the system's pointer back.
#[test]
fn a_scene_without_a_cursor_keeps_the_systems_pointer() {
    let dir = scratch("system", None);
    let mut session = open(&dir);
    let mut play = Play::new(still);
    assert!(
        looks(&mut play, &mut session, 3)
            .iter()
            .all(Option::is_none)
    );

    std::fs::write(dir.join("scene.esn"), scene(Some("vec2(3, 2)"))).unwrap();
    session.sync();
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 3)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![picture(255)]);

    std::fs::write(dir.join("scene.esn"), scene(None)).unwrap();
    session.sync();
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 3)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![PointerLook::System]);
}

/// A picture past the most a pointer may be is said once, and the
/// pointer is the system's.
#[test]
fn a_cursor_too_large_is_said_once_and_leaves_the_systems_pointer() {
    let dir = scratch("large", Some("vec2(3, 2)"));
    let mut session = open(&dir);
    let mut play = Play::new(still);
    looks(&mut play, &mut session, 3);
    play.said();

    std::fs::write(dir.join("scene.esn"), scene(Some("vec2(300, 2)"))).unwrap();
    session.sync();
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 4)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![PointerLook::System]);
    let said = play.said();
    let refusals: Vec<&String> = said
        .iter()
        .filter(|line| line.contains("a pointer is at most 128x128"))
        .collect();
    assert_eq!(refusals.len(), 1, "{said:?}");
    assert!(refusals[0].contains("300x2"), "{said:?}");
}

/// The scene says the pixel its pointer points with; one past the
/// picture is held at its edge.
#[test]
fn the_cursor_hotspot_output_is_the_point_the_pointer_points_with() {
    let hotspot = |at: &str| {
        scene(Some("vec2(3, 2)")).replace(
            "output cursor <- mark.result;\n",
            &format!(
                "output cursor <- mark.result;\nnode point: t.point@1 {{ at = {at}; domain = 1; }}\noutput cursor_hotspot <- point.result;\n"
            ),
        )
    };
    let dir = scratch("hotspot", None);
    std::fs::write(dir.join("scene.esn"), hotspot("vec2(0.0, 1.0)")).unwrap();
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 3)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![pointing(255, (0, 1))]);

    std::fs::write(dir.join("scene.esn"), hotspot("vec2(10.0, -3.0)")).unwrap();
    session.sync();
    let told: Vec<PointerLook> = looks(&mut play, &mut session, 3)
        .into_iter()
        .flatten()
        .collect();
    assert_eq!(told, vec![pointing(255, (2, 0))]);
}

/// The hotspot is two numbers as a stereo sample is; declared before the
/// sound, the sound card still plays the sound.
#[test]
fn the_hotspot_is_never_played_as_the_sound() {
    let dir = scratch("sound", None);
    let scene = scene(Some("vec2(3, 2)")).replace(
        "output cursor <- mark.result;\n",
        "output cursor <- mark.result;\nnode point: t.point@1 { at = vec2(1.0, 1.0); domain = 1; }\noutput cursor_hotspot <- point.result;\nnode tone: t.tone@1 { domain = 16; }\noutput sound <- tone.result;\n",
    );
    std::fs::write(dir.join("scene.esn"), scene).unwrap();
    std::fs::write(
        dir.join("k.ent"),
        format!("{KERNELS}kernel t.tone@1 {{ requires {{ field.write; }} output result: Rows<Vec2>; compute {{ return vec2(0.25, -0.25); }} }}\n"),
    )
    .unwrap();
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let Frame::Shown(shown) = play.frame(&mut session, VIEW) else {
        panic!("a frame was not shown: {}", play.status());
    };
    let (sound, channels) = session.sound_of(&shown.report).expect("a sound");
    assert_eq!(channels, 2);
    assert_eq!(session.runtime().host().domain(sound).unwrap().count(), 16);
    assert_eq!(Some(sound), shown.sound);
}
