//! An edit by declaration changes only those bytes: the rest of the text,
//! comments and layout included, stays as written.

use entelechy_scene::edit::{Edit, StatementKind, Target, edit};

const SCENE: &str = "// The castle on its ground.
use \"kernels/\";

param size: Int = 13;

node ground: voxel.seed@1 { domain = vec3(48, 48, 48); }
// Built on the ground, from the corner.
node castle: castle.build@1 {
    base <- ground.result;
    size = size; // as the scene says
}
node view: voxel.draw@1 { voxels <- castle.result; }

output result <- view.result;
";

fn node(name: &str, text: Option<&str>) -> Edit {
    Edit {
        target: Target::Statement {
            kind: StatementKind::Node,
            name: name.into(),
        },
        text: text.map(str::to_owned),
    }
}

fn member(node: &str, member: &str, text: Option<&str>) -> Edit {
    Edit {
        target: Target::Member {
            node: node.into(),
            member: member.into(),
        },
        text: text.map(str::to_owned),
    }
}

fn statement(kind: StatementKind, name: &str, text: &str) -> Edit {
    Edit {
        target: Target::Statement {
            kind,
            name: name.into(),
        },
        text: Some(text.into()),
    }
}

#[test]
fn a_node_replaced_changes_its_bytes_and_nothing_else() {
    let edited = edit(
        SCENE,
        &[node(
            "view",
            Some("node view: voxel.draw@2 { voxels <- castle.result; }"),
        )],
    )
    .unwrap();
    assert_eq!(edited, SCENE.replace("voxel.draw@1", "voxel.draw@2"));
}

#[test]
fn a_node_not_written_is_added_after_the_last_node() {
    let edited = edit(
        SCENE,
        &[node(
            "moat",
            Some("  node moat: castle.moat@1 { base <- castle.result; }  "),
        )],
    )
    .unwrap();
    assert_eq!(
        edited,
        SCENE.replace(
            "node view: voxel.draw@1 { voxels <- castle.result; }\n",
            "node view: voxel.draw@1 { voxels <- castle.result; }\nnode moat: castle.moat@1 { base <- castle.result; }\n",
        )
    );
}

#[test]
fn a_statement_of_several_lines_stands_apart_by_a_blank_line() {
    let moat = "node moat: castle.moat@1 {\n    base <- castle.result;\n}";
    let edited = edit(SCENE, &[node("moat", Some(moat))]).unwrap();
    assert_eq!(
        edited,
        SCENE.replace(
            "node view: voxel.draw@1 { voxels <- castle.result; }\n",
            &format!("node view: voxel.draw@1 {{ voxels <- castle.result; }}\n\n{moat}\n"),
        )
    );
    // After a node of several lines, a one-line node stands apart too.
    let scene = SCENE.replace("node view: voxel.draw@1 { voxels <- castle.result; }\n", "");
    let edited = edit(
        &scene,
        &[node(
            "wall",
            Some("node wall: castle.wall@1 { base <- castle.result; }"),
        )],
    )
    .unwrap();
    assert!(edited.contains("}\n\nnode wall: castle.wall@1"), "{edited}");
}

#[test]
fn a_node_removed_between_blank_lines_leaves_one() {
    let scene = "use \"kernels/\";\n\nnode a: t.a@1 {\n    x = 1;\n}\n\nnode b: t.b@1 {\n    y <- a.result;\n}\n\noutput r <- b.result;\n";
    let edited = edit(scene, &[node("a", None)]).unwrap();
    assert_eq!(
        edited,
        "use \"kernels/\";\n\nnode b: t.b@1 {\n    y <- a.result;\n}\n\noutput r <- b.result;\n"
    );
}

#[test]
fn a_node_removed_takes_its_lines_and_the_comment_above_it() {
    let edited = edit(SCENE, &[node("castle", None)]).unwrap();
    assert_eq!(
        edited,
        SCENE.replace(
            "// Built on the ground, from the corner.\nnode castle: castle.build@1 {\n    base <- ground.result;\n    size = size; // as the scene says\n}\n",
            "",
        )
    );
}

#[test]
fn members_are_replaced_added_and_removed_as_their_block_is_laid_out() {
    // One line: the added member goes on it.
    let edited = edit(SCENE, &[member("view", "zoom", Some("zoom = 0.5;"))]).unwrap();
    assert!(
        edited.contains("node view: voxel.draw@1 { voxels <- castle.result; zoom = 0.5; }\n"),
        "{edited}"
    );
    // Own lines: an added member gets one, indented alike; a replaced one
    // keeps its trailing comment.
    let edited = edit(
        SCENE,
        &[
            member("castle", "wall", Some("wall = 7;")),
            member("castle", "size", Some("size = 21;")),
        ],
    )
    .unwrap();
    assert!(
        edited.contains(
            "node castle: castle.build@1 {\n    base <- ground.result;\n    size = 21; // as the scene says\n    wall = 7;\n}\n"
        ),
        "{edited}"
    );
    // Removed from its own line, the line and its comment go; from a
    // shared line, the member and the space before it.
    let edited = edit(
        SCENE,
        &[
            member("castle", "size", None),
            member("ground", "domain", None),
        ],
    )
    .unwrap();
    assert!(
        edited.contains("node castle: castle.build@1 {\n    base <- ground.result;\n}\n"),
        "{edited}"
    );
    assert!(
        edited.contains("node ground: voxel.seed@1 { }\n"),
        "{edited}"
    );
    // An empty block takes a first member between its braces.
    let edited = edit(&edited, &[member("ground", "domain", Some("domain = 8;"))]).unwrap();
    assert!(
        edited.contains("node ground: voxel.seed@1 { domain = 8; }\n"),
        "{edited}"
    );
}

#[test]
fn the_first_of_a_kind_goes_after_the_kinds_before_it() {
    // An input goes before the params: after the `use`, a blank line.
    let edited = edit(
        SCENE,
        &[statement(
            StatementKind::Input,
            "font",
            "input font: Grid<Float>;",
        )],
    )
    .unwrap();
    assert!(
        edited.contains("use \"kernels/\";\n\ninput font: Grid<Float>;\n\nparam size"),
        "{edited}"
    );
    // Another param goes after the existing one.
    let edited = edit(
        SCENE,
        &[statement(
            StatementKind::Param,
            "wall",
            "param wall: Int = 5;",
        )],
    )
    .unwrap();
    assert!(
        edited.contains("param size: Int = 13;\nparam wall: Int = 5;\n"),
        "{edited}"
    );
}

#[test]
fn edits_go_in_order_each_on_what_the_last_left() {
    let edited = edit(
        SCENE,
        &[
            node("moat", Some("node moat: castle.moat@1 { }")),
            member("moat", "base", Some("base <- castle.result;")),
            statement(
                StatementKind::Output,
                "result",
                "output result <- moat.result;",
            ),
        ],
    )
    .unwrap();
    assert!(
        edited.contains(
            "node moat: castle.moat@1 { base <- castle.result; }\n\noutput result <- moat.result;\n"
        ),
        "{edited}"
    );
}

#[test]
fn an_edit_that_does_not_say_what_it_names_is_refused() {
    let refused = |edits: &[Edit]| edit(SCENE, edits).unwrap_err().to_string();
    // The text declares another node.
    assert!(
        refused(&[node("view", Some("node other: voxel.draw@1 { }"))])
            .contains("declares node `other`")
    );
    // Two statements for one.
    assert!(
        refused(&[node(
            "view",
            Some("node view: voxel.draw@1 { } node extra: voxel.draw@1 { }")
        )])
        .contains("2 statements")
    );
    // A member that closes its block to write another node.
    assert!(
        refused(&[member(
            "view",
            "zoom",
            Some("zoom = 1; } node evil: a.b@1 {")
        )])
        .contains("not one member")
    );
    // A member that sets something else.
    assert!(refused(&[member("view", "zoom", Some("scale = 1;"))]).contains("sets `scale`"));
    // Nothing to remove, no such node; the index names the edit.
    let error = edit(
        SCENE,
        &[
            member("view", "zoom", Some("zoom = 1;")),
            node("moat", None),
        ],
    )
    .unwrap_err();
    assert_eq!(error.index, 1);
    assert!(
        error.message.contains("no node `moat` to remove"),
        "{error}"
    );
    assert!(refused(&[member("moat", "base", None)]).contains("there is no node `moat`"));
    // A Kernel written in place has no members to edit.
    let inline = "node k: kernel { output result: Rows<Int>; compute { return 0; } }\n";
    let error = edit(inline, &[member("k", "domain", Some("domain = 4;"))]).unwrap_err();
    assert!(error.message.contains("replace the node whole"), "{error}");
}

#[test]
fn a_fragment_ending_in_a_comment_ends_its_line_before_what_followed() {
    let text = "param a: Float = 1.0; param b: Float = 2.0;\nnode k: t.c@1 { domain = 4; }\nnode e: t.c@1 {}\n";
    let edited = edit(
        text,
        &[
            statement(StatementKind::Param, "a", "param a: Float = 3.0; // three"),
            member("k", "from", Some("from = 5.0; // five")),
            member("e", "from", Some("from = 6.0; // six")),
        ],
    )
    .unwrap();
    assert_eq!(
        edited,
        "param a: Float = 3.0; // three\nparam b: Float = 2.0;\nnode k: t.c@1 { domain = 4; from = 5.0; // five\n}\nnode e: t.c@1 { from = 6.0; // six\n}\n"
    );
    // Where nothing follows on the line, the comment stays on it.
    let edited = edit(
        "param a: Float = 1.0;\n",
        &[statement(
            StatementKind::Param,
            "a",
            "param a: Float = 3.0; // three",
        )],
    )
    .unwrap();
    assert_eq!(edited, "param a: Float = 3.0; // three\n");
    // Its comment takes the place of the one that ended the line, a
    // statement's and a member's alike; without one, the old one stays.
    let edited = edit(
        "param a: Float = 1.0;   // one\nnode k: t.c@1 {\n    from = 1.0; // one\n}\n",
        &[
            statement(
                StatementKind::Param,
                "a",
                "param a: Float = 3.0;   // three",
            ),
            member("k", "from", Some("from = 3.0; // three")),
            statement(StatementKind::Param, "a", "param a: Float = 4.0;"),
        ],
    )
    .unwrap();
    assert_eq!(
        edited,
        "param a: Float = 4.0;   // three\nnode k: t.c@1 {\n    from = 3.0; // three\n}\n"
    );
}

/// A text beginning with comment lines takes the place of the comment
/// lines right above the statement it replaces; without them, those stay.
#[test]
fn a_fragment_beginning_with_comments_takes_the_place_of_those_above() {
    let text = "// Kept apart.\n\n/// Old.\n/// Older.\nfn f() -> Float { return 1.0; }\n/// Its own.\nfn g() -> Float { return 2.0; }\n";
    let edited = edit(
        text,
        &[
            statement(
                StatementKind::Fn,
                "f",
                "/// New.\nfn f() -> Float { return 3.0; }",
            ),
            statement(StatementKind::Fn, "g", "fn g() -> Float { return 4.0; }"),
        ],
    )
    .unwrap();
    assert_eq!(
        edited,
        "// Kept apart.\n\n/// New.\nfn f() -> Float { return 3.0; }\n/// Its own.\nfn g() -> Float { return 4.0; }\n"
    );
    // Sent again as it is, it stays one.
    let again = edit(
        &edited,
        &[statement(
            StatementKind::Fn,
            "f",
            "/// New.\nfn f() -> Float { return 3.0; }",
        )],
    )
    .unwrap();
    assert_eq!(again, edited);
}

#[test]
fn a_requirement_is_added_after_the_params_and_replaced_by_name() {
    let edited = edit(
        SCENE,
        &[statement(
            StatementKind::Require,
            "small",
            "require small: size <= 64;",
        )],
    )
    .unwrap();
    assert_eq!(
        edited,
        SCENE.replace(
            "param size: Int = 13;\n",
            "param size: Int = 13;\n\nrequire small: size <= 64;\n"
        )
    );
    let again = edit(
        &edited,
        &[statement(
            StatementKind::Require,
            "small",
            "require small: size <= 32;",
        )],
    )
    .unwrap();
    assert!(again.contains("require small: size <= 32;\n"), "{again}");
}
