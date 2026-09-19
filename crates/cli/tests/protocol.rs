use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value as Json, json};

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-protocol-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/life");
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    for entry in std::fs::read_dir(example.join("kernels"))
        .unwrap()
        .flatten()
    {
        std::fs::copy(entry.path(), dir.join("kernels").join(entry.file_name())).unwrap();
    }
    std::fs::copy(example.join("scene.osn"), dir.join("scene.osn")).unwrap();
    dir
}

/// Sends requests one per line and returns the responses in order.
fn talk(scene: &Path, requests: &[Json]) -> Vec<Json> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("serve")
        .arg(scene)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    let mut lines = stdout.lines();
    let mut responses = Vec::new();
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        let line = lines.next().unwrap().unwrap();
        responses.push(serde_json::from_str(&line).unwrap());
    }
    drop(stdin);
    child.wait().unwrap();
    responses
}

const BRIGHT_SHOW: &str = "kernel life.show@1 { requires { field.read; field.write; } input cells: Image; output result: Image; compute { let c = sample(cells); return if c.r > 0.5 { vec4(1.0, 1.0, 1.0, 1.0) } else { vec4(0.0, 0.0, 0.0, 1.0) }; } }";

#[test]
fn an_agent_can_inspect_prepare_publish_evaluate_and_read() {
    let dir = scratch();
    let scene = dir.join("scene.osn");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "status" }),
            json!({ "id": 2, "op": "kernels" }),
            json!({ "id": 3, "op": "graph" }),
            json!({ "id": 4, "op": "prepare", "source": "kernel x.y@1 {\n  requires { field.write; }\n  output result: Image;\n  compute { return nope; }\n}" }),
            json!({ "id": 5, "op": "publish", "source": BRIGHT_SHOW, "expect": 999 }),
            json!({ "id": 6, "op": "kernel", "id": "life.show@1" }),
            json!({ "id": 7, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 8, "op": "read", "format": "rgba32f" }),
            json!({ "id": 9, "op": "param", "name": "nothing", "value": [1, 2] }),
            json!({ "id": 10, "op": "compose", "text": "node broken: {" }),
            json!({ "id": 11, "op": "evaluate", "time": 0.1, "pointer": [40.0, 40.0], "down": true, "keys": [65] }),
            json!({ "id": 12, "op": "retire", "id": "life.show@1" }),
            json!({ "id": 13, "op": "evaluate", "time": 0.2 }),
            json!({ "id": 14, "op": "nonsense" }),
            json!({ "id": 15, "op": "quit" }),
        ],
    );
    let ok = |index: usize| -> &Json {
        let response = &responses[index];
        assert!(response.get("error").is_none(), "{response}");
        &response["ok"]
    };
    assert_eq!(responses[0]["id"], 1);
    assert_eq!(ok(0)["backend"], "cpu");
    assert_eq!(ok(0)["nodes"], 3);

    let kernels = ok(1)["kernels"].as_array().unwrap();
    let step = kernels.iter().find(|k| k["id"] == "life.step@1").unwrap();
    assert_eq!(step["inputs"][0]["name"], "previous");
    assert_eq!(step["params"][0]["name"], "brush");
    assert_eq!(step["params"][0]["default"], 2.0);
    assert!(
        step["requires"]
            .as_array()
            .unwrap()
            .iter()
            .any(|r| r == "input.read")
    );

    let graph = ok(2);
    assert_eq!(graph["nodes"][1]["id"], "life");
    assert_eq!(graph["nodes"][1]["inputs"]["previous"]["kind"], "previous");
    assert_eq!(
        graph["nodes"][1]["inputs"]["previous"]["initial"],
        "seed.result"
    );
    assert_eq!(graph["outputs"][0]["source"], "show.result");

    let error = &responses[3]["error"];
    assert_eq!(error["line"], 4);
    assert_eq!(error["column"], 20);
    assert!(
        error["message"]
            .as_str()
            .unwrap()
            .contains("`nope` is not defined"),
        "{error}"
    );

    let stale = &responses[4]["error"];
    assert!(
        stale["message"]
            .as_str()
            .unwrap()
            .contains("registry holds"),
        "{stale}"
    );

    let show = ok(5);
    assert_eq!(show["id"], "life.show@1");
    let generation = show["generation"].as_u64().unwrap();
    assert!(show["source"].as_str().unwrap().contains("life.show@1"));

    let report = ok(6);
    assert_eq!(report["executed"].as_array().unwrap().len(), 3);
    assert_eq!(report["committed"][0][0], "life");
    assert_eq!(report["outputs"][0]["width"], 160);

    let read = ok(7);
    assert_eq!(read["format"], "rgba32f");
    let data = read["data"].as_str().unwrap();
    assert_eq!(data.len(), (160 * 120 * 16usize).div_ceil(3) * 4);

    assert_eq!(ok(8)["value"], json!([1.0, 2.0]));

    let events = ok(9)["events"].as_array().unwrap();
    assert_eq!(events[0]["kind"], "scene_rejected");
    assert!(events[0]["line"].is_number(), "{events:?}");

    let report = ok(10);
    assert!(
        report["executed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n == "life"),
        "{report}"
    );

    assert_eq!(ok(11)["retired"], true);
    let error = &responses[12]["error"]["message"];
    assert!(error.as_str().unwrap().contains("life.show@1"), "{error}");
    assert!(
        responses[13]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("unknown op")
    );
    assert_eq!(ok(14)["bye"], true);
    let _ = generation;
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn fields_come_in_through_write_and_buttons_reach_kernels() {
    // Its own directory: `scratch` is shared by the tests of this binary.
    let dir = std::env::temp_dir().join(format!("orla-protocol-write-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("k.orla"),
        "kernel t.zero@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }
kernel t.count@1 { requires { field.read; field.write; } input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) + 1.0; } }
kernel t.stir@1 { requires { field.write; input.read; } output result: Rows<Float>; compute { return (if button(1) { 1.0 } else { 0.0 }) + wheel().y * 10.0; } }",
    )
    .unwrap();
    let scene = dir.join("scene.osn");
    std::fs::write(
        &scene,
        "use \"k.orla\";
node seed: t.zero@1 { domain = 2; }
node count: t.count@1 { previous <~ count.result from seed.result; }
node stir: t.stir@1 { domain = 1; }
output result <- count.result;
output stirred <- stir.result;
",
    )
    .unwrap();
    let words = |values: &[f32]| {
        orla::protocol::base64(
            &values
                .iter()
                .flat_map(|v| v.to_bits().to_le_bytes())
                .collect::<Vec<u8>>(),
        )
    };
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "read", "node": "count" }),
            json!({ "id": 3, "op": "write", "node": "count", "data": words(&[40.0, 50.0]) }),
            json!({ "id": 4, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 5, "op": "read", "node": "count" }),
            json!({ "id": 6, "op": "write", "node": "seed", "data": words(&[1.0, 2.0]) }),
            json!({ "id": 7, "op": "write", "node": "count", "data": words(&[1.0]) }),
            json!({ "id": 8, "op": "evaluate", "time": 0.2, "buttons": [1], "wheel": [0.0, -1.0] }),
            json!({ "id": 9, "op": "read", "output": "stirred" }),
            json!({ "id": 10, "op": "quit" }),
        ],
    );
    let floats = |index: usize| -> Vec<f32> {
        let response = &responses[index];
        assert!(response.get("error").is_none(), "{response}");
        let data = response["ok"]["data"].as_str().unwrap();
        orla::protocol::unbase64(data)
            .unwrap()
            .chunks_exact(4)
            .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect()
    };
    assert_eq!(floats(1), vec![1.0, 1.0]);
    assert_eq!(responses[2]["ok"]["words"], 2, "{}", responses[2]);
    assert_eq!(
        floats(4),
        vec![41.0, 51.0],
        "the next frame counts on from what was written"
    );
    let message = |index: usize| {
        responses[index]["error"]["message"]
            .as_str()
            .unwrap()
            .to_owned()
    };
    assert!(message(5).contains("not state"), "{}", message(5));
    assert!(message(6).contains("holds 2 words"), "{}", message(6));
    assert_eq!(
        floats(8),
        vec![-9.0],
        "the right button and the wheel reached the Kernel"
    );
}
