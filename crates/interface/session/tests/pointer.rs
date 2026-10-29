//! The Host's devices a scene gives its outputs to, the pointer above all
//! (`entelechy_session::play::PointerLook`): the picture a scene gives `to
//! pointer` is the system's pointer in every interface, pointing with the
//! pixel it gives `to pointer_tip`, else its middle; an interface is told
//! when it changes and only then, one too large is said once and leaves
//! the system's pointer, and a scene without one keeps the system's. A
//! device takes only what is given it, and one the Host has not is refused.

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
            "node mark: t.mark@1 {{ domain = {domain}; }}\noutput view <- view.result;\noutput cursor <- mark.result to pointer;\n"
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
fn what_the_scene_gives_pointer_tip_is_the_point_the_pointer_points_with() {
    let hotspot = |at: &str| {
        scene(Some("vec2(3, 2)")).replace(
            "output cursor <- mark.result to pointer;\n",
            &format!(
                "output cursor <- mark.result to pointer;\nnode point: t.point@1 {{ at = {at}; domain = 1; }}\noutput cursor_hotspot <- point.result to pointer_tip;\n"
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

/// A device takes what the scene gives it and nothing else: rows of two
/// floats given no device are not played, and the window shows the first
/// picture given no device, though the pointer's is declared first.
#[test]
fn a_device_takes_only_what_the_scene_gives_it() {
    let dir = scratch("given", None);
    std::fs::write(
        dir.join("k.ent"),
        format!("{KERNELS}kernel t.tone@1 {{ requires {{ field.write; }} output result: Rows<Vec2>; compute {{ return vec2(0.25, -0.25); }} }}\n"),
    )
    .unwrap();
    let scene = |sound: &str| {
        format!(
            "use \"k.ent\";\nnode mark: t.mark@1 {{ domain = vec2(3, 2); }}\nnode view: t.view@1 {{ domain = vec2(4, 4); }}\nnode point: t.point@1 {{ at = vec2(1.0, 1.0); domain = 1; }}\nnode tone: t.tone@1 {{ domain = 16; }}\noutput cursor <- mark.result to pointer;\noutput tip <- point.result to pointer_tip;\noutput view <- view.result;\noutput tone <- tone.result{sound};\n"
        )
    };
    std::fs::write(dir.join("scene.esn"), scene("")).unwrap();
    let mut session = open(&dir);
    let mut play = Play::new(still);
    let Frame::Shown(shown) = play.frame(&mut session, VIEW) else {
        panic!("a frame was not shown: {}", play.status());
    };
    assert_eq!(shown.sound, None, "rows given no device are not played");
    let picture = session.runtime().host().domain(shown.picture).unwrap();
    assert_eq!(
        picture.count(),
        16,
        "the window shows the view, not the pointer"
    );

    std::fs::write(dir.join("scene.esn"), scene(" to sound")).unwrap();
    session.sync();
    let Frame::Shown(shown) = play.frame(&mut session, VIEW) else {
        panic!("a frame was not shown: {}", play.status());
    };
    let (sound, channels) = session.sound_of(&shown.report).expect("a sound");
    assert_eq!(channels, 2);
    assert_eq!(session.runtime().host().domain(sound).unwrap().count(), 16);
    assert_eq!(Some(sound), shown.sound);
}

/// A device the Host does not have is refused as the scene loads, naming
/// the nearest it has.
#[test]
fn a_device_the_host_does_not_have_is_refused() {
    let dir = scratch("unknown", None);
    std::fs::write(
        dir.join("scene.esn"),
        "use \"k.ent\";\nnode view: t.view@1 { domain = vec2(4, 4); }\noutput view <- view.result to pionter;\n",
    )
    .unwrap();
    let mut session = Session::open(&dir.join("scene.esn"), None, 1 << 20);
    let said: Vec<String> = session.sync().iter().map(ToString::to_string).collect();
    assert!(session.graph().is_none(), "{said:?}");
    assert!(
        said.iter()
            .any(|line| line.contains("no device `pionter`") && line.contains("pointer")),
        "{said:?}"
    );
}

/// A check, before anything runs, holds each output to the structure its
/// device takes, and refuses a device the Host has not.
#[test]
fn a_check_holds_each_output_to_its_devices_structure() {
    let dir = scratch("checked", None);
    std::fs::write(
        dir.join("scene.esn"),
        "use \"k.ent\";\nnode view: t.view@1 { domain = vec2(4, 4); }\noutput view <- view.result;\noutput noise <- view.result to sound;\noutput copied <- view.result to clipboard;\n",
    )
    .unwrap();
    let checked = entelechy_session::check_document(
        &entelechy_session::Disk::default(),
        &dir.join("scene.esn"),
        false,
    );
    let problems: Vec<&String> = checked
        .lines
        .iter()
        .filter(|(p, _)| *p)
        .map(|(_, l)| l)
        .collect();
    assert_eq!(problems.len(), 2, "{problems:?}");
    assert!(
        problems
            .iter()
            .any(|l| l.contains("output `noise`: `to sound` takes a frame's samples")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|l| l.contains("output `copied`: `to clipboard` takes text")),
        "{problems:?}"
    );
    assert!(
        problems.iter().all(|l| l.contains("scene.esn:")),
        "each said where it is written: {problems:?}"
    );
}
