//! Edits of a document by what it declares change only those bytes: the
//! rest of the text, comments and layout, is as it was written.

use orla_scene::edit::{Edit, StatementKind, Target, edit};

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
    // One line: the member added goes on it.
    let edited = edit(SCENE, &[member("view", "zoom", Some("zoom = 0.5;"))]).unwrap();
    assert!(
        edited.contains("node view: voxel.draw@1 { voxels <- castle.result; zoom = 0.5; }\n"),
        "{edited}"
    );
    // Lines of their own: the member added gets one, indented alike;
    // one replaced keeps the comment after it.
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
    // Removed from a line of its own, the line and its comment go; from a
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
    // An input comes before the params: after the `use`, a blank line.
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
    // Another param goes after the one there.
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
    // Nothing to remove, no such node; the index says which edit.
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
