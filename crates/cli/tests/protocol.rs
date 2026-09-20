use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value as Json, json};

fn scratch() -> PathBuf {
    scratch_in("")
}

/// A copy of the Life example in a directory of its own: tests in this
/// binary run in parallel, so each takes a different `name`.
fn scratch_in(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-protocol{name}-{}", std::process::id()));
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
kernel t.stir@1 { requires { field.write; input.read; } output result: Rows<Float>; compute { return (if button(1) { 1.0 } else { 0.0 }) + wheel().y * 10.0; } }
struct Pair { a: Float; b: Float; }
fn twice(x: Float) -> Float { return x * 2.0; }",
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
            // A Kernel published through the protocol sees the scene's declarations.
            json!({ "id": 10, "op": "publish", "source": "kernel t.pair@1 { requires { field.write; } output result: Rows<Pair>; compute { return Pair { a: twice(1.0), b: 2.0 }; } }" }),
            json!({ "id": 11, "op": "prepare", "source": "struct Own { x: Float; }\nkernel t.own@1 { requires { field.write; } output result: Rows<Own>; compute { return Own { x: 1.0 }; } }" }),
            json!({ "id": 12, "op": "prepare", "source": "kernel t.a@1 { requires { field.write; } output result: Rows<Float>; compute { return 1.0; } }\nkernel t.b@1 { requires { field.write; } output result: Rows<Float>; compute { return 2.0; } }" }),
            // A Field file goes out and comes back through the protocol.
            json!({ "id": 13, "op": "read", "node": "count", "format": "field" }),
            json!({ "id": 14, "op": "quit" }),
        ],
    );
    let file = orla::protocol::unbase64(responses[12]["ok"]["data"].as_str().unwrap()).unwrap();
    assert!(
        file.starts_with(b"ORLAFIELD 1\nRows 2 1 1 f\n"),
        "{:?}",
        String::from_utf8_lossy(&file[..24])
    );
    let again = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "write", "node": "count", "format": "field", "data": orla::protocol::base64(&file) }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 4, "op": "read", "node": "count" }),
            json!({ "id": 5, "op": "write", "node": "count", "format": "field", "data": orla::protocol::base64(&orla_runtime::file::encode(&"f".parse().unwrap(), orla_runtime::Domain::Rows(3), &[0, 0, 0])) }),
            json!({ "id": 6, "op": "quit" }),
        ],
    );
    let back: Vec<f32> = orla::protocol::unbase64(again[3]["ok"]["data"].as_str().unwrap())
        .unwrap()
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect();
    assert_eq!(
        back,
        vec![43.0, 53.0],
        "the saved count came back and counted on"
    );
    assert!(
        again[4]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("but `count.result` is"),
        "{}",
        again[4]
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
    assert_eq!(responses[9]["ok"]["id"], "t.pair@1", "{}", responses[9]);
    assert_eq!(responses[10]["ok"]["id"], "t.own@1", "{}", responses[10]);
    assert!(message(11).contains("one kernel"), "{}", message(11));
}

#[test]
fn the_protocol_serves_sockets_and_a_quit_ends_only_its_connection() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-sockets");
    let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("serve")
        .arg(dir.join("scene.osn"))
        .args(["--listen", "127.0.0.1:0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = BufReader::new(child.stderr.take().unwrap());
    let port = stderr
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("listening on 127.0.0.1:")
                .map(str::to_owned)
        })
        .expect("the server names its port");
    let talk = |requests: &[Json]| -> Vec<Json> {
        let stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
        let mut writer = stream.try_clone().unwrap();
        let mut reader = BufReader::new(stream);
        let mut responses = Vec::new();
        for request in requests {
            writeln!(writer, "{request}").unwrap();
            let mut line = String::new();
            reader.read_line(&mut line).unwrap();
            responses.push(serde_json::from_str(&line).unwrap());
        }
        let mut rest = String::new();
        reader.read_line(&mut rest).unwrap();
        assert!(rest.is_empty(), "the connection closed after quit: {rest}");
        responses
    };
    let first = talk(&[
        json!({ "id": 1, "op": "status" }),
        json!({ "id": 2, "op": "evaluate", "time": 0.0 }),
        json!({ "id": 3, "op": "quit" }),
    ]);
    assert_eq!(first[0]["ok"]["nodes"], 3, "{}", first[0]);
    assert_eq!(first[1]["ok"]["executed"].as_array().unwrap().len(), 3);
    assert_eq!(first[2]["ok"]["bye"], true);
    // The server outlived that client's quit; a second client is served.
    let second = talk(&[
        json!({ "id": 4, "op": "status" }),
        json!({ "id": 5, "op": "quit" }),
    ]);
    assert_eq!(second[0]["ok"]["kernels"], 4, "{}", second[0]);
    // A quit on standard input ends the process.
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 6, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_browser_speaks_the_protocol_over_a_websocket() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-websocket");
    let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("serve")
        .arg(dir.join("scene.osn"))
        .args(["--listen", "127.0.0.1:0"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let stderr = BufReader::new(child.stderr.take().unwrap());
    let port = stderr
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("listening on 127.0.0.1:")
                .map(str::to_owned)
        })
        .expect("the server names its port");
    // The handshake of RFC 6455, with its own example key.
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    write!(
        stream,
        "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut head = String::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
        head.push_str(&line);
    }
    assert!(head.starts_with("HTTP/1.1 101"), "{head}");
    assert!(
        head.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo="),
        "{head}"
    );
    // A masked text frame per request, as a browser sends them; an
    // unmasked one per answer.
    let send = |stream: &mut TcpStream, text: &str| {
        let mask = [0x12u8, 0x34, 0x56, 0x78];
        let bytes = text.as_bytes();
        let mut frame = vec![0x81u8];
        if bytes.len() < 126 {
            frame.push(0x80 | bytes.len() as u8);
        } else {
            frame.push(0x80 | 126);
            frame.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
        }
        frame.extend_from_slice(&mask);
        frame.extend(bytes.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
        stream.write_all(&frame).unwrap();
    };
    let receive = |reader: &mut BufReader<TcpStream>| -> (u8, Vec<u8>) {
        let mut head = [0u8; 2];
        reader.read_exact(&mut head).unwrap();
        let mut length = (head[1] & 0x7f) as usize;
        if length == 126 {
            let mut ext = [0u8; 2];
            reader.read_exact(&mut ext).unwrap();
            length = u16::from_be_bytes(ext) as usize;
        } else if length == 127 {
            let mut ext = [0u8; 8];
            reader.read_exact(&mut ext).unwrap();
            length = u64::from_be_bytes(ext) as usize;
        }
        let mut payload = vec![0u8; length];
        reader.read_exact(&mut payload).unwrap();
        (head[0] & 0x0f, payload)
    };
    send(&mut stream, &json!({ "id": 1, "op": "status" }).to_string());
    let (opcode, payload) = receive(&mut reader);
    assert_eq!(opcode, 1);
    let status: Json = serde_json::from_slice(&payload).unwrap();
    assert_eq!(status["id"], 1);
    assert_eq!(status["ok"]["nodes"], 3, "{status}");
    send(
        &mut stream,
        &json!({ "id": 2, "op": "evaluate", "time": 0.0 }).to_string(),
    );
    let (_, payload) = receive(&mut reader);
    let evaluated: Json = serde_json::from_slice(&payload).unwrap();
    assert_eq!(evaluated["ok"]["executed"].as_array().unwrap().len(), 3);
    // A read's answer is longer than a short frame allows.
    send(
        &mut stream,
        &json!({ "id": 3, "op": "read", "format": "ppm" }).to_string(),
    );
    let (_, payload) = receive(&mut reader);
    let read: Json = serde_json::from_slice(&payload).unwrap();
    assert_eq!(read["ok"]["format"], "ppm", "{read}");
    assert!(read["ok"]["data"].as_str().unwrap().len() > 125);
    // A ping is answered with a pong carrying the payload.
    stream
        .write_all(&[0x89, 0x82, 1, 2, 3, 4, b'h' ^ 1, b'i' ^ 2])
        .unwrap();
    let (opcode, payload) = receive(&mut reader);
    assert_eq!((opcode, payload.as_slice()), (0xA, &b"hi"[..]));
    // A quit answers, then the server closes the connection.
    send(&mut stream, &json!({ "id": 4, "op": "quit" }).to_string());
    let (_, payload) = receive(&mut reader);
    let bye: Json = serde_json::from_slice(&payload).unwrap();
    assert_eq!(bye["ok"]["bye"], true);
    let (opcode, _) = receive(&mut reader);
    assert_eq!(opcode, 0x8, "a close frame follows the quit");
    // A plain HTTP request is told what this is.
    let mut plain = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    write!(plain, "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
    let mut answer = String::new();
    plain.read_to_string(&mut answer).unwrap();
    assert!(answer.starts_with("HTTP/1.1 426"), "{answer}");
    assert!(answer.contains("creation protocol"), "{answer}");
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 6, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}
