use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value as Json, json};

/// A copy of the Life example in a directory of its own: tests in this
/// binary run in parallel, so each takes a different `name`.
fn scratch_in(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-protocol{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/life");
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
    talk_with(scene, &[], requests)
}

/// [`talk`] to a session started with `flags` after the scene.
fn talk_with(scene: &Path, flags: &[&str], requests: &[Json]) -> Vec<Json> {
    let mut child = Command::new(env!("CARGO_BIN_EXE_orla"))
        .arg("serve")
        .arg(scene)
        .args(flags)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let stdout = BufReader::new(child.stdout.take().unwrap());
    // What the child says on stderr, drained as it comes so a talkative
    // child never blocks on a full pipe, kept for a diagnosis.
    let mut stderr = child.stderr.take().unwrap();
    let said = std::thread::spawn(move || {
        let mut said = String::new();
        let _ = std::io::Read::read_to_string(&mut stderr, &mut said);
        said
    });
    let mut lines = stdout.lines();
    let mut responses = Vec::new();
    for request in requests {
        writeln!(stdin, "{request}").unwrap();
        stdin.flush().unwrap();
        // A missing answer is the served child gone: say how, with what
        // it said, rather than just where.
        let Some(line) = lines.next() else {
            drop(stdin);
            let status = child.wait().unwrap();
            let said = said.join().unwrap_or_default();
            panic!(
                "`orla serve` answered nothing to {request}: it exited with {status}, having said:\n{said}"
            );
        };
        let line = line.unwrap();
        responses.push(serde_json::from_str(&line).unwrap());
    }
    drop(stdin);
    child.wait().unwrap();
    let _ = said.join();
    responses
}

const BRIGHT_SHOW: &str = "kernel life.show@1 { requires { field.read; field.write; } input cells: Image; output result: Image; compute { let c = sample(cells); return if c.r > 0.5 { vec4(1.0, 1.0, 1.0, 1.0) } else { vec4(0.0, 0.0, 0.0, 1.0) }; } }";

/// A failed update leaves the running system alive, whatever the
/// Kernel: one chained or nested past what a document may hold is
/// refused with where, by a live session, which then goes on evaluating
/// — rather than taking the process down with a stack its compiler ran
/// out of. An `if` in an arm of an `if` two hundred deep is read once.
#[test]
fn a_kernel_too_deep_to_compile_is_refused_and_the_session_lives_on() {
    let dir = scratch_in("-deep");
    let scene = dir.join("scene.osn");
    let kernel = |body: String| {
        format!(
            "kernel deep.k@1 {{ requires {{ field.write; }} output result: Rows<Int>; compute {{ {body} }} }}"
        )
    };
    let chained = kernel(format!(
        "return element(){};",
        (0..20_000)
            .map(|k| format!(" + {}", k % 7))
            .collect::<String>()
    ));
    let nested = kernel(format!(
        "return {}element(){};",
        "(".repeat(60_000),
        ")".repeat(60_000)
    ));
    let branches = 100;
    let ifs = kernel(format!(
        "let x = element(); return {}x{};",
        (0..branches)
            .map(|k| format!("if x > {k} {{ "))
            .collect::<String>(),
        " } else { 0 }".repeat(branches)
    ));
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "publish", "source": chained }),
            json!({ "id": 2, "op": "publish", "source": nested }),
            json!({ "id": 3, "op": "publish", "source": ifs }),
            json!({ "id": 4, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 5, "op": "status" }),
            json!({ "id": 6, "op": "quit" }),
        ],
    );
    for index in [0, 1] {
        let message = responses[index]["error"]["message"]
            .as_str()
            .unwrap_or_default();
        assert!(message.contains("deep"), "{}", responses[index]);
    }
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    assert!(responses[3].get("error").is_none(), "{}", responses[3]);
    assert_eq!(responses[4]["ok"]["nodes"], 3, "{}", responses[4]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_agent_can_inspect_prepare_publish_evaluate_and_read() {
    let dir = scratch_in("-inspect");
    let scene = dir.join("scene.osn");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "status" }),
            json!({ "id": 2, "op": "kernels" }),
            json!({ "id": 3, "op": "graph" }),
            json!({ "id": 4, "op": "prepare", "source": "kernel x.y@1 {\n  requires { field.write; }\n  output result: Image;\n  compute { return nope; }\n}" }),
            json!({ "id": 5, "op": "publish", "source": BRIGHT_SHOW, "expect": 999 }),
            json!({ "id": 6, "op": "kernel", "kernel": "life.show@1" }),
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
    assert_eq!(ok(0)["backend"], "jit");
    assert_eq!(ok(0)["nodes"], 3);

    let kernels = ok(1)["kernels"].as_array().unwrap();
    let step = kernels.iter().find(|k| k["id"] == "life.step@1").unwrap();
    assert_eq!(step["inputs"][0]["name"], "previous");
    // A port says what it holds by the name the Kernel gave it, not by
    // the struct's number in the module.
    assert_eq!(step["inputs"][0]["type"], "Grid<Cell>");
    assert_eq!(step["outputs"][0]["type"], "Grid<Cell>");
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

    // A param the scene does not declare reaches nothing, and is refused.
    assert!(
        responses[8]["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains("`nothing` is not a param the scene declares")),
        "{}",
        responses[8]
    );

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
        orla_formats::base64::base64(
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
    let file =
        orla_formats::base64::unbase64(responses[12]["ok"]["data"].as_str().unwrap()).unwrap();
    assert!(
        file.starts_with(b"ORLAFIELD 1\nRows 2 1 1 f\n"),
        "{:?}",
        String::from_utf8_lossy(&file[..24])
    );
    let again = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "write", "node": "count", "format": "field", "data": orla_formats::base64::base64(&file) }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 4, "op": "read", "node": "count" }),
            json!({ "id": 5, "op": "write", "node": "count", "format": "field", "data": orla_formats::base64::base64(&orla_formats::file::encode(&"f".parse().unwrap(), orla_runtime::Domain::Rows(3), &[0, 0, 0])) }),
            json!({ "id": 6, "op": "quit" }),
        ],
    );
    let back: Vec<f32> = orla_formats::base64::unbase64(again[3]["ok"]["data"].as_str().unwrap())
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
        orla_formats::base64::unbase64(data)
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

/// A listening server a test started, killed when the test ends, however
/// it ends: one that listens does not stop when its standard input
/// closes, so a failing test would leave it running.
struct Server(std::process::Child);

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
fn the_protocol_serves_sockets_and_a_quit_ends_only_its_connection() {
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-sockets");
    let mut child = Server(
        Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("serve")
            .arg(dir.join("scene.osn"))
            .args(["--listen", "127.0.0.1:0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = BufReader::new(child.0.stderr.take().unwrap());
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
    // Three of the scene's own and the two the runtime publishes itself,
    // the rasterizer and its cutout.
    assert_eq!(second[0]["ok"]["kernels"], 5, "{}", second[0]);
    // A quit on standard input ends the process.
    let mut stdin = child.0.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 6, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.0.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_browser_speaks_the_protocol_over_a_websocket() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-websocket");
    let mut child = Server(
        Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("serve")
            .arg(dir.join("scene.osn"))
            .args(["--listen", "127.0.0.1:0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = BufReader::new(child.0.stderr.take().unwrap());
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
    let mut stdin = child.0.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 6, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.0.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A peer that says it will send more than a request may hold is told so
/// and let go; the server, and every other connection, lives on.
#[test]
fn a_peer_claiming_an_endless_request_is_refused_and_the_server_lives() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-endless");
    let mut child = Server(
        Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("serve")
            .arg(dir.join("scene.osn"))
            .args(["--listen", "127.0.0.1:0"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = BufReader::new(child.0.stderr.take().unwrap());
    let port = stderr
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("listening on 127.0.0.1:")
                .map(str::to_owned)
        })
        .expect("the server names its port");
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    write!(
        stream,
        "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
    )
    .unwrap();
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    // A text frame that says it carries 2^63 bytes.
    let mut frame = vec![0x81u8, 0x80 | 127];
    frame.extend_from_slice(&(u64::MAX >> 1).to_be_bytes());
    frame.extend_from_slice(&[1, 2, 3, 4]);
    stream.write_all(&frame).unwrap();
    let mut answer = Vec::new();
    reader.read_to_end(&mut answer).unwrap();
    let answer = String::from_utf8_lossy(&answer);
    assert!(answer.contains("at most"), "{answer}");
    // Another client is served, by the same process.
    let stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
    let mut writer = stream.try_clone().unwrap();
    let mut reader = BufReader::new(stream);
    writeln!(writer, "{}", json!({ "id": 1, "op": "status" })).unwrap();
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let status: Json = serde_json::from_str(&line).unwrap();
    assert_eq!(status["ok"]["nodes"], 3, "{status}");
    let mut stdin = child.0.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 2, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.0.wait().unwrap().success());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_headless_session_takes_a_window_for_window_shaped_scenes() {
    let dir = scratch_in("-window");
    std::fs::write(
        dir.join("window.osn"),
        "node fill: kernel { requires { field.write; input.read; } output result: Image; domain = window(); compute { let w = window(); return vec4(w.x / 100.0, scale(), 0.0, 1.0); } }
output result <- fill.result;
",
    )
    .unwrap();
    let responses = talk(
        &dir.join("window.osn"),
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "window", "size": [8, 4], "scale": 2 }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 4, "op": "read", "format": "rgba32f" }),
            json!({ "id": 5, "op": "evaluate", "time": 0.2, "window": [6, 3] }),
            json!({ "id": 6, "op": "window", "size": null }),
            json!({ "id": 7, "op": "evaluate", "time": 0.3 }),
            json!({ "id": 8, "op": "window", "size": [0, 4] }),
        ],
    );
    assert!(
        responses[0]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no window"),
        "{}",
        responses[0]
    );
    assert_eq!(responses[1]["ok"]["window"], json!([8.0, 4.0]));
    assert_eq!(responses[2]["ok"]["outputs"][0]["width"], 8);
    assert_eq!(responses[2]["ok"]["outputs"][0]["height"], 4);
    let bytes = {
        let data = responses[3]["ok"]["data"].as_str().unwrap();
        orla_formats::base64::unbase64(data).unwrap()
    };
    let first = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    let second = f32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
    assert_eq!(
        (first, second),
        (0.08, 2.0),
        "window() and scale() reach the Kernel"
    );
    // A window on the request wins over the session's; none, and the
    // scene cannot be sized again.
    assert_eq!(responses[4]["ok"]["outputs"][0]["width"], 6);
    assert_eq!(responses[5]["ok"]["window"], Json::Null);
    assert!(
        responses[6]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("no window")
    );
    assert!(
        responses[7]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at least one")
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_trace_shows_one_element_of_a_node_and_reports_carry_timings() {
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/chip8/scene.osn");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0, "window": [128, 64] }),
            json!({ "id": 2, "op": "trace", "node": "machine", "element": 0, "window": [128, 64] }),
            json!({ "id": 3, "op": "trace", "node": "view", "at": [4, 4], "window": [128, 64] }),
            json!({ "id": 4, "op": "trace", "node": "nowhere", "element": 0 }),
            json!({ "id": 5, "op": "trace", "node": "machine" }),
            // Element 2^32 is past the end, not element 0 again.
            json!({ "id": 6, "op": "trace", "node": "machine", "element": 4_294_967_296u64, "window": [128, 64] }),
        ],
    );
    let timings = responses[0]["ok"]["timings"].as_array().unwrap();
    assert!(
        timings
            .iter()
            .any(|t| t["node"] == "machine" && t["millis"].is_number()),
        "{timings:?}"
    );
    let trace = &responses[1]["ok"];
    assert_eq!(trace["kernel"], "chip8.step@1");
    assert_eq!(trace["at"], json!([0, 0, 0]));
    let bindings = trace["bindings"].as_array().unwrap();
    let pc = bindings.iter().find(|b| b["name"] == "pc").unwrap();
    assert_eq!(pc["type"], "Int");
    assert!(pc["value"].as_i64().unwrap() >= 512, "{pc}");
    assert!(
        bindings
            .iter()
            .any(|b| b["name"] == "drawn" && b["type"] == "Draw" && b["value"].is_object())
    );
    assert!(
        bindings.iter().any(|b| b["ran"] == false),
        "a branch not taken is marked"
    );
    assert!(
        trace["calls"].is_array() && trace["calls_truncated"] == false,
        "{trace}"
    );
    assert_eq!(trace["results"][0]["type"], "Int");
    let view = &responses[2]["ok"];
    assert_eq!(view["at"], json!([4, 4, 0]));
    assert!(
        view["results"][0]["value"].as_array().unwrap().len() == 4,
        "{view}"
    );
    assert!(
        responses[3]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("nowhere")
    );
    assert!(
        responses[4]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("element")
    );
    assert!(responses[5]["error"].is_object(), "{}", responses[5]);
}

#[test]
fn help_answers_the_builtins_and_the_forms_as_json() {
    let dir = scratch_in("-help");
    let responses = talk(
        &dir.join("scene.osn"),
        &[
            json!({ "id": 1, "op": "help" }),
            json!({ "id": 2, "op": "help", "name": "mix" }),
            json!({ "id": 3, "op": "help", "name": "mixx" }),
            json!({ "id": 4, "op": "help", "name": "each" }),
        ],
    );
    let all = responses[0]["ok"]["builtins"].as_array().unwrap();
    assert!(all.len() > 50);
    assert!(
        all.iter()
            .any(|b| b["name"] == "put" && b["requires"] == "field.write")
    );
    let mix = &responses[1]["ok"];
    assert_eq!(mix["kind"], "math");
    assert!(
        mix["signatures"]
            .as_array()
            .unwrap()
            .iter()
            .any(|s| s == "mix(a: Vec3, b: Vec3, t: Float) -> Vec3")
    );
    assert!(mix["requires"].is_null());
    assert!(
        responses[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mix")
    );
    // The forms of the language answer the same way.
    let forms = responses[0]["ok"]["forms"].as_array().unwrap();
    assert!(forms.len() > 15);
    assert!(
        forms
            .iter()
            .any(|f| f["name"] == "scatter" && f["kind"] == "kernel")
    );
    assert!(
        forms
            .iter()
            .any(|f| f["name"] == "fill" && f["kind"] == "graph")
    );
    let each = &responses[3]["ok"];
    assert_eq!(each["kind"], "expression");
    assert_eq!(each["shape"], "[each n { body }]");
    assert!(each["doc"].as_str().unwrap().contains("element by element"));
}

/// The scanline example's `agent.jsonl` is the record of an agent
/// creating a capability while the scene runs: it looks around, proposes
/// a Kernel with a mistake, reads the positioned diagnostic, repairs it,
/// publishes, composes the Kernel into the scene, observes the picture
/// change, turns the effect up, and fetches its own Kernel back. This
/// runs that record against a copy of the example and holds every step
/// to what the example's README says happens.
#[test]
fn an_agent_creates_a_capability_while_the_scene_runs() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/scanline");
    let dir = std::env::temp_dir().join(format!("orla-protocol-agent-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::copy(
        example.join("kernels/picture.orla"),
        dir.join("kernels/picture.orla"),
    )
    .unwrap();
    std::fs::copy(example.join("scene.osn"), dir.join("scene.osn")).unwrap();
    let script = std::fs::read_to_string(example.join("agent.jsonl")).unwrap();
    let requests: Vec<Json> = script
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(requests.len(), 18);
    let responses = talk(&dir.join("scene.osn"), &requests);
    let ok = |index: usize| -> &Json {
        assert!(
            responses[index].get("ok").is_some(),
            "request {} failed: {}",
            requests[index]["id"],
            responses[index]
        );
        &responses[index]["ok"]
    };
    // Looking around: the scene, the Kernels there, a builtin's contract.
    assert_eq!(ok(0)["nodes"], 1);
    assert!(
        ok(1)["kernels"]
            .as_array()
            .unwrap()
            .iter()
            .any(|k| k["id"] == "picture.draw@1")
    );
    assert_eq!(ok(2)["name"], "coord");
    // The first proposal is refused where it is wrong: a Float `&` an Int
    // on line 9, at the expression.
    let error = &responses[5]["error"];
    assert_eq!(error["message"], "cannot apply `&` to Float and Int");
    assert_eq!(
        (error["line"].as_u64(), error["column"].as_u64()),
        (Some(9), Some(20))
    );
    // The repair prepares, unpublished, then publishes as a new Kernel.
    assert_eq!(ok(7)["id"], "scanline.effect@1");
    assert_eq!(ok(7)["current"], Json::Null);
    assert_eq!(ok(8)["id"], "scanline.effect@1");
    assert!(ok(8)["generation"].as_u64().unwrap() >= 1);
    // The scene composed with the new Kernel loads, twice.
    for index in [10, 13] {
        let events = ok(index)["events"].as_array().unwrap();
        assert!(
            events.iter().any(|e| e["kind"] == "scene_loaded")
                && !events.iter().any(|e| e["kind"] == "scene_rejected"),
            "{}",
            responses[index]
        );
    }
    // The picture: even rows as they were, odd rows darkened by the
    // strength, 0.5 then 0.8.
    let pixels = |index: usize| -> (usize, usize, Vec<f32>) {
        let image = ok(index);
        let bytes = orla_formats::base64::unbase64(image["data"].as_str().unwrap()).unwrap();
        let floats = bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        (
            image["width"].as_u64().unwrap() as usize,
            image["height"].as_u64().unwrap() as usize,
            floats,
        )
    };
    let (w, h, before) = pixels(4);
    let (_, _, half) = pixels(12);
    let (_, _, strong) = pixels(15);
    assert_eq!((w, h), (160, 120));
    for y in 0..h {
        for x in 0..w {
            for k in 0..3 {
                let at = (y * w + x) * 4 + k;
                let (dim_half, dim_strong): (f32, f32) =
                    if y % 2 == 1 { (0.5, 0.2) } else { (1.0, 1.0) };
                assert!(
                    (half[at] - before[at] * dim_half).abs() < 1e-5,
                    "({x}, {y})"
                );
                assert!(
                    (strong[at] - before[at] * dim_strong).abs() < 1e-5,
                    "({x}, {y})"
                );
            }
        }
    }
    // The Kernel comes back as it was published, the repaired line in it.
    assert!(
        ok(16)["source"]
            .as_str()
            .unwrap()
            .contains("icoord().y % 2 == 1")
    );
    assert_eq!(ok(17)["bye"], true);
}

/// The migrate example's `agent.jsonl`: an agent changes what a running
/// tally carries, publishing the new Kernel and a migration, and the
/// count goes on; then it drops the migration and the old layout comes
/// back, and the tally restarts from its seed, reported.
#[test]
fn an_agent_changes_a_running_states_layout_and_the_state_is_carried_over() {
    let example = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/migrate");
    let dir = std::env::temp_dir().join(format!("orla-protocol-migrate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("kernels")).unwrap();
    std::fs::copy(
        example.join("kernels/tally.orla"),
        dir.join("kernels/tally.orla"),
    )
    .unwrap();
    std::fs::copy(example.join("scene.osn"), dir.join("scene.osn")).unwrap();
    let requests: Vec<Json> = std::fs::read_to_string(example.join("agent.jsonl"))
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let responses = talk(&dir.join("scene.osn"), &requests);
    let ok = |index: usize| -> &Json {
        assert!(
            responses[index].get("ok").is_some(),
            "request {} failed: {}",
            requests[index]["id"],
            responses[index]
        );
        &responses[index]["ok"]
    };
    let state = |index: usize| -> Vec<i32> {
        orla_formats::base64::unbase64(ok(index)["data"].as_str().unwrap())
            .unwrap()
            .chunks(4)
            .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };
    assert_eq!(state(4), vec![3], "three evaluations counted");
    assert_eq!(ok(5)["id"], "tally.upgrade@1");
    assert_eq!(ok(6)["id"], "tally.step@1");
    // Composed with `migrate`, the next evaluation carries the state over.
    assert_eq!(
        ok(9)["migrated"],
        json!([{ "node": "count", "port": "previous", "kernel": "tally.upgrade@1" }])
    );
    assert_eq!(ok(9)["reset"], json!([]));
    assert_eq!(
        state(10),
        vec![4, 8],
        "the count went on, the double from it"
    );
    assert_eq!(ok(11)["migrated"], json!([]));
    assert_eq!(state(12), vec![5, 10]);
    // Without a migration, the old layout back: the state restarts.
    assert_eq!(ok(15)["migrated"], json!([]));
    let reset = ok(15)["reset"].as_array().unwrap();
    assert_eq!(reset.len(), 1, "{}", responses[15]);
    assert_eq!(reset[0]["node"], "count");
    let reason = reset[0]["reason"].as_str().unwrap();
    assert!(
        reason.contains("an element of") && reason.contains("integer"),
        "{reason}"
    );
    assert_eq!(state(16), vec![1]);
    assert_eq!(ok(17)["bye"], true);
}

/// A run that spans many frames is one request: `advance` steps the
/// scene, stops when a value says to, saves what the agent asked for,
/// and reports what happened. An agent training a network asks once
/// instead of once a frame.
#[test]
fn advance_runs_many_frames_and_stops_when_a_value_says_to() {
    let dir = std::env::temp_dir().join(format!("orla-protocol-advance-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "node zero: Rows<Int> { domain = 1; }\n\
         node tally: kernel {\n\
             requires { field.read; field.write; }\n\
             input previous: Rows<Int> <~ tally.result from zero.result;\n\
             output result: Rows<Int>;\n\
             domain = 1;\n\
             compute { return at(previous, 0) + 1; }\n\
         }\n\
         node view: kernel {\n\
             requires { field.read; field.write; }\n\
             input count: Rows<Int> <- tally.result;\n\
             output result: Image;\n\
             domain = vec2(4, 4);\n\
             compute { let c = float(at(count, 0)) / 255.0; return vec4(c, c, c, 1.0); }\n\
         }\n\
         output result <- view.result;\n\
         output count <- tally.result;\n",
    )
    .unwrap();
    let kept = dir.join("count.field");
    let allow = format!("file.write={}", dir.display());
    let responses = talk_with(
        &dir.join("scene.osn"),
        &["--allow", &allow],
        &[
            json!({ "id": 1, "op": "advance", "frames": 500,
                    "until": { "target": "tally.result[0]", "above": 41.5 },
                    "save": { "tally.result": "count.field" } }),
            json!({ "id": 2, "op": "advance", "frames": 3 }),
            json!({ "id": 3, "op": "advance", "frames": 0 }),
            json!({ "id": 4, "op": "advance", "frames": 2,
                    "until": { "target": "tally.result[0]", "over": 9 } }),
        ],
    );
    let run = &responses[0]["ok"];
    assert_eq!(run["stopped"], "until");
    assert_eq!(run["frames"], 42, "one frame per count: {run}");
    assert_eq!(run["value"], 42.0);
    assert_eq!(run["target"], "tally.result[0]");
    assert!(run["millis"].as_f64().unwrap() > 0.0);
    // Only what the run watched had to be materialized: the picture was
    // not asked for, so it did not run.
    let executed = run["report"]["executed"].as_array().unwrap();
    assert!(executed.iter().any(|node| node == "tally"), "{run}");
    assert!(!executed.iter().any(|node| node == "view"), "{run}");
    // What it saved is a Field of the Host's own format.
    let saved = run["saved"].as_array().unwrap();
    assert_eq!(saved.len(), 1);
    assert!(kept.exists(), "{saved:?}");
    assert!(std::fs::read(&kept).unwrap().len() > 8);
    // Without an `until`, it runs the count it was given, and the whole
    // scene runs because nothing was named.
    let plain = &responses[1]["ok"];
    assert_eq!(plain["stopped"], "frames");
    assert_eq!(plain["frames"], 3);
    assert!(plain["value"].is_null());
    assert!(
        plain["report"]["executed"]
            .as_array()
            .unwrap()
            .iter()
            .any(|node| node == "view")
    );
    // A count of zero and a test nobody wrote are refused by name.
    assert!(
        responses[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("frames"),
        "{:?}",
        responses[2]
    );
    assert!(
        responses[3]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("`below`, `above` or `equals`"),
        "{:?}",
        responses[3]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// An agent can measure what it changed: the same frames several times,
/// the least of the runs, and what each node cost, as `orla bench`
/// reports from a terminal. The clock advances while it measures, as
/// `advance` advances it.
#[test]
fn an_agent_can_measure_the_scene_it_is_holding() {
    let dir = scratch_in("-measure");
    let responses = talk(
        &dir.join("scene.osn"),
        &[
            json!({ "id": 1, "op": "bench", "frames": 4, "repeat": 2, "nodes": true }),
            json!({ "id": 2, "op": "bench", "frames": 0 }),
            json!({ "id": 3, "op": "bench", "frames": 100000, "repeat": 4 }),
            // A product that passes the end of 64 bits is still too many.
            json!({ "id": 4, "op": "bench", "frames": 1001u64, "repeat": 17_285_760_180_958_600_815u64 }),
            json!({ "id": 5, "op": "quit" }),
        ],
    );
    let measured = &responses[0]["ok"];
    assert_eq!(measured["frames"], 4, "{}", responses[0]);
    assert_eq!(measured["repeat"], 2);
    assert_eq!(measured["ran"], true);
    assert!(measured["best_ms"].as_f64().unwrap() > 0.0);
    assert!(measured["median_ms"].as_f64().unwrap() >= measured["best_ms"].as_f64().unwrap());
    // Life's two nodes, the costlier first.
    let nodes = measured["nodes"].as_array().unwrap();
    assert_eq!(nodes.len(), 2, "{measured}");
    assert!(nodes[0]["ms"].as_f64().unwrap() >= nodes[1]["ms"].as_f64().unwrap());

    assert!(
        responses[1]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("whole number"),
        "{}",
        responses[1]
    );
    assert!(
        responses[2]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at most"),
        "{}",
        responses[2]
    );
    assert!(
        responses[3]["error"]["message"]
            .as_str()
            .unwrap()
            .contains("at most"),
        "{}",
        responses[3]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A name that is not a node is refused where it is written, rather
/// than accepted by `pin` and reported a request later as a read that
/// says to pin it — which is what sent one reader looking for a fault
/// in the materializing.
#[test]
fn a_name_that_is_not_a_node_is_refused_by_the_op_that_takes_it() {
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/mesh/scene.osn");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "pin", "node": "vertices" }),
            json!({ "id": 2, "op": "pin", "node": "subdivided.vertice" }),
            json!({ "id": 3, "op": "pin", "node": "subdivided.vertices" }),
            json!({ "id": 4, "op": "unpin", "node": "nowhere" }),
            json!({ "id": 5, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 6, "op": "read", "node": "verticez" }),
            json!({ "id": 7, "op": "read", "node": "subdivided.vertices" }),
            json!({ "id": 8, "op": "quit" }),
        ],
    );
    let message = |at: usize| -> String {
        responses[at]["error"]["message"]
            .as_str()
            .unwrap_or_default()
            .to_owned()
    };
    assert!(
        message(0).contains("`vertices` is not a node of this scene"),
        "{}",
        message(0)
    );
    // A near miss says which node was meant.
    assert!(
        message(1).contains("did you mean `subdivided.vertices`?"),
        "{}",
        message(1)
    );
    assert!(responses[2]["ok"].is_object(), "{:?}", responses[2]);
    assert!(message(3).contains("not a node"), "{}", message(3));
    assert!(message(5).contains("not a node"), "{}", message(5));
    // The node that is one reads, having been pinned.
    assert!(responses[6]["ok"].is_object(), "{:?}", responses[6]);
}

#[test]
fn a_matrix_or_a_struct_parameter_crosses_the_protocol_as_arrays() {
    // JSON has arrays and no types: the parameter a value is given to
    // takes it in its own shape, and says what it is when it cannot.
    let scene = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../examples/cube/scene.osn");
    let replies = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "param", "name": "lens", "value": [[3, 0, 0, 0], [0, 3, 0, 0], [0, 0, 1, -1], [0, 0, 0, 0]] }),
            json!({ "id": 2, "op": "param", "name": "style", "value": [[0.9, 0.9, 0.9], [0.8, 0.1, 0.1], 2] }),
            json!({ "id": 3, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 4, "op": "param", "name": "lens", "value": [1, 2, 3] }),
            json!({ "id": 5, "op": "graph" }),
        ],
    );
    assert_eq!(
        replies[0]["ok"]["value"],
        json!([
            [3.0, 0.0, 0.0, 0.0],
            [0.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, -1.0],
            [0.0, 0.0, 0.0, 0.0]
        ])
    );
    assert!(replies[1].get("ok").is_some(), "{}", replies[1]);
    assert!(replies[2].get("ok").is_some(), "{}", replies[2]);
    let message = replies[3]["error"].to_string();
    assert!(message.contains("param `lens` is a Mat4"), "{message}");
    let params = replies[4]["ok"]["params"].as_array().unwrap();
    let style = params.iter().find(|p| p["name"] == "style").unwrap();
    assert_eq!(style["type"], "Style");
    assert_eq!(style["value"][2], json!(2.0));
}

/// An agent saves by the allowance the person running the session gave,
/// as a scene's outputs are written: with none, or outside it, nothing
/// is written, and nothing runs either.
#[test]
fn advance_saves_only_where_writes_are_allowed() {
    let dir = scratch_in("-save-allowed");
    let saves = dir.join("saves");
    std::fs::create_dir_all(&saves).unwrap();
    let outside = dir.join("outside.field");
    let inside = saves.join("cells.field");
    let escape = saves.join("../outside.field");
    let request = |id: u32, path: &Path| {
        json!({ "id": id, "op": "advance", "frames": 1,
                "save": { "life.result": path.display().to_string() } })
    };
    let closed = talk(&dir.join("scene.osn"), &[request(1, &inside)]);
    let why = closed[0]["error"]["message"].as_str().unwrap_or_default();
    assert!(why.contains("--allow file.write"), "{}", closed[0]);
    assert!(!inside.exists());

    let allow = format!("file.write={}", saves.display());
    let open = talk_with(
        &dir.join("scene.osn"),
        &["--allow", &allow],
        &[
            request(1, &outside),
            request(2, &escape),
            request(3, &inside),
        ],
    );
    for refused in &open[..2] {
        let why = refused["error"]["message"].as_str().unwrap_or_default();
        assert!(why.contains("outside every directory"), "{refused}");
    }
    assert!(!outside.exists());
    assert_eq!(
        open[2]["ok"]["saved"].as_array().unwrap().len(),
        1,
        "{}",
        open[2]
    );
    assert!(inside.exists());
}

/// A composed scene is checked before it is written: a text that would
/// be rejected, by its syntax or by a Kernel it defines, is answered as
/// a sync answers it and never reaches the disk, so the document an
/// agent reads back is the scene running. With `expect`, a writer that
/// read an older version is refused.
#[test]
fn compose_checks_before_it_writes_and_against_the_version_read() {
    let dir = scratch_in("-compose");
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let inline_broken = original.replace(
        "node show: life.show@1 {",
        "node bad: kernel { requires { field.write; } output result: Image; compute { return nope; } }\nnode show: life.show@1 {",
    );
    let grown = original.replace("brush = 2.0;", "brush = 3.0;");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "document" }),
            json!({ "id": 2, "op": "compose", "text": "node broken: {" }),
            json!({ "id": 3, "op": "compose", "text": inline_broken }),
            json!({ "id": 4, "op": "compose", "text": grown, "expect": "0000000000000000" }),
            json!({ "id": 5, "op": "document" }),
            json!({ "id": 6, "op": "evaluate", "time": 0.0 }),
        ],
    );
    let version = responses[0]["ok"]["version"].as_str().unwrap().to_owned();
    assert_eq!(responses[0]["ok"]["text"], original);
    for (index, what) in [(1, "syntax"), (2, "an inline Kernel")] {
        let answer = &responses[index]["ok"];
        assert_eq!(
            answer["events"][0]["kind"], "scene_rejected",
            "{what}: {answer}"
        );
        assert_eq!(answer["version"], version, "{what}");
    }
    assert!(
        responses[2]["ok"]["events"][0]["message"]
            .as_str()
            .unwrap_or_default()
            .contains("`nope` is not defined"),
        "{}",
        responses[2]
    );
    let stale = &responses[3]["error"];
    assert!(
        stale["message"]
            .as_str()
            .unwrap()
            .contains("changed since it was read"),
        "{stale}"
    );
    assert_eq!(stale["version"], version);
    // Nothing that was refused reached the disk.
    assert_eq!(responses[4]["ok"]["text"], original);
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), original);
    assert!(responses[5].get("error").is_none(), "{}", responses[5]);

    // Against the version it read, the change is taken.
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "compose", "text": grown, "expect": version }),
            json!({ "id": 2, "op": "document" }),
        ],
    );
    let taken = &responses[0]["ok"];
    assert!(
        taken["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "scene_loaded"),
        "{taken}"
    );
    assert_ne!(taken["version"], json!(version));
    assert_eq!(responses[1]["ok"]["version"], taken["version"]);
    assert_eq!(responses[1]["ok"]["text"], grown);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A served scene is composed by an agent and read back through `read`,
/// so what its inputs read is held to its documents' directories and the
/// ones the person allowed: a file elsewhere is refused by name, and read
/// once `--allow file.read` names its directory.
#[test]
fn a_served_scene_reads_beside_its_documents_or_where_allowed() {
    let base = std::env::temp_dir().join(format!("orla-protocol-reads-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (dir, elsewhere) = (base.join("scene"), base.join("elsewhere"));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::create_dir_all(&elsewhere).unwrap();
    std::fs::write(dir.join("beside.bin"), b"ab").unwrap();
    std::fs::write(elsewhere.join("secret.bin"), b"xyz").unwrap();
    let scene = dir.join("scene.osn");
    let text = |path: &str| {
        format!(
            "input data: Rows<Int> = bytes \"{path}\";\n\
             node copy: kernel {{ requires {{ field.read; field.write; }} input bytes: Rows<Int> <- data; output result: Rows<Int>; domain = 1; compute {{ return at(bytes, 0); }} }}\n\
             output result <- copy.result;\n"
        )
    };
    std::fs::write(&scene, text("beside.bin")).unwrap();
    let secret = elsewhere.join("secret.bin").display().to_string();
    let escape = "../elsewhere/secret.bin";
    let kinds = |answer: &Json| -> Vec<String> {
        answer["ok"]["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|event| {
                format!(
                    "{} {}",
                    event["kind"].as_str().unwrap_or_default(),
                    event["message"]
                        .as_str()
                        .or(event["error"].as_str())
                        .unwrap_or_default()
                )
            })
            .collect()
    };
    let closed = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "compose", "text": text(&secret) }),
            json!({ "id": 2, "op": "compose", "text": text(escape) }),
            json!({ "id": 3, "op": "compose", "text": text("beside.bin") }),
        ],
    );
    for refused in &closed[..2] {
        let said = kinds(refused).join("\n");
        assert!(
            said.contains("outside the directories the scene's documents are in or use"),
            "{said}"
        );
    }
    assert!(
        kinds(&closed[2]).iter().any(|k| k.starts_with("loaded")),
        "{:?}",
        kinds(&closed[2])
    );

    let allow = format!("file.read={}", elsewhere.display());
    let open = talk_with(
        &scene,
        &["--allow", &allow],
        &[json!({ "id": 1, "op": "compose", "text": text(&secret) })],
    );
    assert!(
        kinds(&open[0]).iter().any(|k| k.starts_with("loaded")),
        "{:?}",
        kinds(&open[0])
    );
    let _ = std::fs::remove_dir_all(&base);
}

/// Whether a session's reads are held is decided when it is opened: once
/// its scene has been read, holding them is refused, so what may be read
/// never depends on when it was asked, or on who spoke first.
#[test]
fn reads_are_held_from_the_start_or_not_at_all() {
    let dir = scratch_in("-held-late");
    let mut session = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    session.sync();
    let refused = session.restrict_reads().unwrap_err();
    assert!(
        refused.contains("from before the scene is first read"),
        "{refused}"
    );
    let mut held = orla_session::Session::open(&dir.join("scene.osn"), None, 1 << 20);
    held.restrict_reads().unwrap();
    held.sync();
    let _ = std::fs::remove_dir_all(&dir);
}

/// Any web page the person visits can reach a server on this machine. A
/// page elsewhere is refused its WebSocket unless `--allow-origin` names
/// it, and an HTTP request that is no upgrade, a form's or a fetch's, is
/// answered and closed rather than read as request lines, so its body
/// composes nothing.
#[test]
fn a_page_elsewhere_can_neither_open_a_socket_nor_post_requests() {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpStream;
    let dir = scratch_in("-origins");
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let mut child = Server(
        Command::new(env!("CARGO_BIN_EXE_orla"))
            .arg("serve")
            .arg(&scene)
            .args([
                "--listen",
                "127.0.0.1:0",
                "--allow-origin",
                "https://studio.example",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let stderr = BufReader::new(child.0.stderr.take().unwrap());
    let port = stderr
        .lines()
        .map_while(Result::ok)
        .find_map(|line| {
            line.strip_prefix("listening on 127.0.0.1:")
                .map(str::to_owned)
        })
        .expect("the server names its port");
    let answer = |request: String| -> String {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}")).unwrap();
        stream.write_all(request.as_bytes()).unwrap();
        let mut head = [0u8; 12];
        stream.read_exact(&mut head).unwrap();
        String::from_utf8_lossy(&head).into_owned()
    };
    let upgrade = |origin: &str| {
        format!(
            "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nOrigin: {origin}\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
    };
    assert_eq!(answer(upgrade("https://evil.example")), "HTTP/1.1 403");
    assert_eq!(
        answer(upgrade("http://localhost.evil.example")),
        "HTTP/1.1 403"
    );
    assert_eq!(answer(upgrade("null")), "HTTP/1.1 101");
    assert_eq!(answer(upgrade("http://localhost:5173")), "HTTP/1.1 101");
    assert_eq!(answer(upgrade("https://studio.example")), "HTTP/1.1 101");
    let compose = json!({ "id": 1, "op": "compose", "text": "node broken: {" }).to_string();
    let post = format!(
        "POST / HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{compose}\n",
        compose.len() + 1
    );
    assert_eq!(answer(post), "HTTP/1.1 400");
    // Nothing the page sent reached the session.
    let mut stdin = child.0.stdin.take().unwrap();
    writeln!(stdin, "{}", json!({ "id": 2, "op": "quit" })).unwrap();
    drop(stdin);
    assert!(child.0.wait().unwrap().success());
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), original);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A server started with `--grant` gives Kernels those capabilities and
/// no others: `status` says which, and a Kernel an agent publishes that
/// requires another is refused with what it lacks, the one running kept.
#[test]
fn a_served_kernel_gets_only_what_the_server_grants() {
    let dir = scratch_in("-grants");
    let reads_clock = BRIGHT_SHOW.replace(
        "requires { field.read; field.write; }",
        "requires { field.read; field.write; time.read; }",
    );
    let responses = talk_with(
        &dir.join("scene.osn"),
        &["--grant", "field.read,field.write", "--grant", "input.read"],
        &[
            json!({ "id": 1, "op": "status" }),
            json!({ "id": 2, "op": "publish", "source": reads_clock, "expect": "any" }),
            json!({ "id": 3, "op": "kernel", "kernel": "life.show@1" }),
            json!({ "id": 4, "op": "publish", "source": BRIGHT_SHOW, "expect": "any" }),
        ],
    );
    assert_eq!(
        responses[0]["ok"]["grants"],
        json!(["field.read", "field.write", "input.read"])
    );
    let why = responses[1]["error"]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        why.contains("requires time.read, which this session does not grant"),
        "{}",
        responses[1]
    );
    assert_eq!(responses[2]["ok"]["generation"], 3, "{}", responses[2]);
    assert!(responses[3].get("error").is_none(), "{}", responses[3]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The world's history over the wire: a publish is a change with where it
/// came from, a refused one is an attempt that moves nothing, an old
/// generation's source can still be read, and frames count within a
/// revision.
#[test]
fn history_says_what_changed_the_world_and_what_was_refused() {
    let dir = scratch_in("-history");
    // The example's own show, its dead cells a little lighter; `Cells`
    // is the scene's, which a published Kernel inherits.
    let lighter = "kernel life.show@1 { requires { field.read; field.write; } input cells: Cells; output result: Image; compute { let cell = sample(cells); let dead = vec3(0.2, 0.2, 0.25); let color = if cell.alive { vec3(1.0, 0.85, 0.45) } else { dead }; return vec4(color.x, color.y, color.z, 1.0); } }";
    let broken = lighter.replace("sample(cells)", "nope");
    let responses = talk(
        &dir.join("scene.osn"),
        &[
            json!({ "id": 1, "op": "status" }),
            json!({ "id": 2, "op": "publish", "source": lighter, "expect": "any" }),
            json!({ "id": 3, "op": "publish", "source": broken, "expect": "any" }),
            json!({ "id": 4, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 5, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 6, "op": "status" }),
            json!({ "id": 7, "op": "history", "since": 0 }),
            json!({ "id": 8, "op": "kernel", "kernel": "life.show@1", "generation": 3 }),
            json!({ "id": 9, "op": "kernel", "kernel": "life.show@1", "generation": 99 }),
        ],
    );
    let ok = |i: usize| -> &Json {
        assert!(responses[i].get("error").is_none(), "{}", responses[i]);
        &responses[i]["ok"]
    };
    let before = ok(0)["revision"].as_u64().unwrap();
    ok(3);
    ok(4);
    // The publish made one revision; the refused one made none; two
    // frames ran in it.
    assert_eq!(ok(5)["revision"], before + 1, "{}", ok(5));
    assert_eq!(ok(5)["frame"], 2);
    let history = ok(6);
    let changes = history["changes"].as_array().unwrap();
    let published = changes.last().unwrap();
    assert_eq!(published["kind"], "published", "{published}");
    assert_eq!(published["kernel"], "life.show@1");
    assert_eq!(published["from"], 3);
    assert_eq!(published["to"], 4);
    assert_eq!(published["origin"], json!({ "protocol": "stdin" }));
    // What loaded the scene came from its documents.
    assert!(
        changes.iter().any(|c| c["kind"] == "published"
            && c["origin"]["document"]
                .as_str()
                .is_some_and(|d| d.ends_with("life.orla"))),
        "{history}"
    );
    let attempt = history["attempts"]
        .as_array()
        .unwrap()
        .last()
        .unwrap()
        .clone();
    assert_eq!(attempt["subject"], "life.show@1", "{attempt}");
    assert!(
        attempt["why"].as_str().unwrap().contains("`nope`"),
        "{attempt}"
    );
    assert_eq!(attempt["revision"], before + 1);
    // The document's own show, generation 3, which the publish replaced.
    let old = ok(7)["source"].as_str().unwrap();
    assert!(
        old.contains("kernel life.show@1") && old.contains("0.07"),
        "{old}"
    );
    let forgotten = responses[8]["error"]["message"].as_str().unwrap();
    assert!(
        forgotten.contains("not one the session remembers"),
        "{forgotten}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A snapshot takes the world as committed: after an evaluation that
/// failed, `read` shows the output it computed and did not commit, and
/// the snapshot shows the state that stands. It carries every Kernel's
/// source, and is written to a file only where writes are allowed.
#[test]
fn a_snapshot_takes_committed_state_and_the_laws_it_ran_under() {
    let dir = std::env::temp_dir().join(format!("orla-protocol-snapshot-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("saves")).unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        "node zero: Rows<Int> { domain = 1; }\n\
         node tally: kernel {\n\
             requires { field.read; field.write; }\n\
             input previous: Rows<Int> <~ tally.result from zero.result;\n\
             output result: Rows<Int>;\n\
             domain = 1;\n\
             compute { return at(previous, 0) + 1; }\n\
         }\n\
         node check: kernel {\n\
             requires { field.read; field.write; }\n\
             input count: Rows<Int> <- tally.result;\n\
             output result: Rows<Int>;\n\
             domain = 1;\n\
             compute { return 100 / (at(count, 0) - 3); }\n\
         }\n\
         output result <- check.result;\n",
    )
    .unwrap();
    let words = |response: &Json| -> Vec<u32> {
        let data = response["data"].as_str().unwrap();
        let bytes = orla_formats::base64::unbase64(data).unwrap();
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let allow = format!("file.write={}", dir.join("saves").display());
    let responses = talk_with(
        &dir.join("scene.osn"),
        &["--allow", &allow],
        &[
            json!({ "id": 1, "op": "pin", "node": "tally" }),
            json!({ "id": 2, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
            // The third count divides by zero: nothing of it commits.
            json!({ "id": 4, "op": "evaluate", "time": 0.2 }),
            json!({ "id": 5, "op": "read", "node": "tally" }),
            json!({ "id": 6, "op": "snapshot" }),
            json!({ "id": 7, "op": "snapshot", "to": "saves/world.json" }),
            json!({ "id": 8, "op": "snapshot", "to": "elsewhere.json" }),
        ],
    );
    assert!(responses[3].get("error").is_some(), "{}", responses[3]);
    // The cache holds the count the failed evaluation computed...
    assert_eq!(words(&responses[4]["ok"]), [3], "{}", responses[4]);
    // ...and the snapshot the count that stands.
    let snapshot = &responses[5]["ok"];
    assert_eq!(snapshot["snapshot"], 1);
    let states = snapshot["states"].as_array().unwrap();
    let tally = states.iter().find(|s| s["node"] == "tally").unwrap();
    assert_eq!(words(tally), [2], "{tally}");
    assert_eq!(tally["shape"], "Rows");
    assert_eq!(tally["extent"], json!([1, 1, 1]));
    let laws = snapshot["laws"].as_array().unwrap();
    assert!(
        laws.iter().any(|l| l["source"]
            .as_str()
            .is_some_and(|s| s.contains("at(previous, 0) + 1"))),
        "{snapshot}"
    );
    let written = &responses[6]["ok"];
    let file = std::fs::read_to_string(written["path"].as_str().unwrap()).unwrap();
    let file: Json = serde_json::from_str(&file).unwrap();
    assert_eq!(file["states"], snapshot["states"]);
    let refused = responses[7]["error"]["message"].as_str().unwrap();
    assert!(refused.contains("outside every directory"), "{refused}");
    assert!(!dir.join("elsewhere.json").exists());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A tally in a library of its own, so a publish can replace it by id,
/// and a check that fails when the count reaches `fail_at`.
fn tally_world(name: &str, fail_at: i32) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("orla-protocol-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("lib.orla"),
        "kernel t.tally@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 1; } }\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("scene.osn"),
        format!(
            "use \"lib.orla\";\n\
             node zero: Rows<Int> {{ domain = 1; }}\n\
             node tally: t.tally@1 {{ previous <~ tally.result from zero.result; domain = 1; }}\n\
             node check: kernel {{\n\
                 requires {{ field.read; field.write; }}\n\
                 input count: Rows<Int> <- tally.result;\n\
                 output result: Rows<Int>;\n\
                 domain = 1;\n\
                 compute {{ return 100 / (at(count, 0) - {fail_at}); }}\n\
             }}\n\
             output result <- check.result;\n"
        ),
    )
    .unwrap();
    dir.join("scene.osn")
}

fn counted(snapshot: &Json) -> u32 {
    let tally = snapshot["states"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["node"] == "tally")
        .unwrap();
    let bytes = orla_formats::base64::unbase64(tally["data"].as_str().unwrap()).unwrap();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// A snapshot's world restored under the laws that run: the trial frame
/// proves it and commits nothing, so the world is the snapshot's and the
/// next frame counts on from it. A restore says which laws, or is
/// refused.
#[test]
fn a_restored_world_is_the_snapshots_and_counts_on_from_it() {
    let scene = tally_world("restore", 1_000_000);
    // The snapshot is taken in a first session and restored in a second:
    // it is data its holder keeps, and needs nothing of the session.
    let first = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 3, "op": "evaluate", "time": 0.2 }),
            json!({ "id": 4, "op": "snapshot" }),
        ],
    );
    let taken = first[3]["ok"].clone();
    assert_eq!(counted(&taken), 3);
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "restore", "snapshot": taken }),
            json!({ "id": 3, "op": "restore", "snapshot": taken, "laws": "current" }),
            json!({ "id": 4, "op": "snapshot" }),
            json!({ "id": 5, "op": "evaluate", "time": 0.3 }),
            json!({ "id": 6, "op": "snapshot" }),
            json!({ "id": 7, "op": "history" }),
        ],
    );
    let why = responses[1]["error"]["message"].as_str().unwrap();
    assert!(why.contains("`laws`"), "{why}");
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    // The trial frame committed nothing: the world is the snapshot's...
    assert_eq!(counted(&responses[3]["ok"]), 3);
    // ...and the next frame is the one after it.
    assert_eq!(counted(&responses[5]["ok"]), 4);
    let changes = responses[6]["ok"]["changes"].as_array().unwrap();
    let restored = changes.iter().find(|c| c["kind"] == "restored").unwrap();
    assert_eq!(restored["laws"], "current");
    assert_eq!(restored["origin"], json!({ "protocol": "stdin" }));
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// A world that cannot run is not restored: its trial frame fails, the
/// state that stood stands, and the attempt is recorded with why.
#[test]
fn a_world_that_cannot_run_is_refused_and_the_world_stands() {
    // The check divides by zero when the count reaches 3.
    let scene = tally_world("refused", 3);
    let ten = orla_formats::base64::base64(&10u32.to_le_bytes());
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 3, "op": "snapshot" }),
            json!({ "id": 4, "op": "write", "node": "tally", "data": ten }),
            json!({ "id": 5, "op": "status" }),
        ],
    );
    let at_two = responses[2]["ok"].clone();
    assert_eq!(counted(&at_two), 2);
    let revision = responses[4]["ok"]["revision"].clone();
    // Restored, the count of 2 would step to 3 and divide by zero.
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "write", "node": "tally", "data": ten }),
            json!({ "id": 3, "op": "restore", "snapshot": at_two, "laws": "current" }),
            json!({ "id": 4, "op": "snapshot" }),
            json!({ "id": 5, "op": "history" }),
        ],
    );
    let why = responses[2]["error"]["message"].as_str().unwrap();
    assert!(why.contains("does not run"), "{why}");
    assert_eq!(
        counted(&responses[3]["ok"]),
        10,
        "the state that stood stands"
    );
    let attempts = responses[4]["ok"]["attempts"].as_array().unwrap();
    assert!(
        attempts.last().unwrap()["subject"]
            .as_str()
            .unwrap()
            .contains("snapshot")
    );
    let _ = revision;
    // With its laws, the same: the law the snapshot would bring back is
    // taken back with the state, and the law that ran runs.
    let faster = "kernel t.tally@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 10; } }";
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "publish", "source": faster, "expect": "any" }),
            json!({ "id": 2, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 3, "op": "restore", "snapshot": at_two, "laws": "snapshot" }),
            json!({ "id": 4, "op": "kernel", "kernel": "t.tally@1" }),
            json!({ "id": 5, "op": "snapshot" }),
        ],
    );
    assert!(responses[2].get("error").is_some(), "{}", responses[2]);
    let source = responses[3]["ok"]["source"].as_str().unwrap();
    assert!(source.contains("+ 10;"), "the law that ran runs: {source}");
    assert_eq!(counted(&responses[4]["ok"]), 10);
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// With its laws, a snapshot brings back the Kernels that ran when it was
/// taken: a tally changed since goes back to what it was, all together
/// with the state, and counts on as it counted then.
#[test]
fn a_world_restored_with_its_laws_runs_as_it_ran() {
    let scene = tally_world("laws", 1_000_000);
    let faster = "kernel t.tally@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 10; } }";
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "snapshot" }),
            json!({ "id": 3, "op": "publish", "source": faster, "expect": "any" }),
            json!({ "id": 4, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 5, "op": "snapshot" }),
        ],
    );
    let at_one = responses[1]["ok"].clone();
    assert_eq!(counted(&at_one), 1);
    assert_eq!(counted(&responses[4]["ok"]), 11, "the faster law ran");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "publish", "source": faster, "expect": "any" }),
            json!({ "id": 2, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 3, "op": "restore", "snapshot": at_one, "laws": "snapshot" }),
            json!({ "id": 4, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 5, "op": "snapshot" }),
            json!({ "id": 6, "op": "kernel", "kernel": "t.tally@1" }),
        ],
    );
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    assert_eq!(
        counted(&responses[4]["ok"]),
        2,
        "one, and one step of the old law"
    );
    let source = responses[5]["ok"]["source"].as_str().unwrap();
    assert!(source.contains("+ 1;"), "{source}");
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// Structure has a past as the laws do: every scene the session loaded
/// is kept by its version, `document` reads an old one, and composing it
/// again, against the version that stands, rolls the structure back.
#[test]
fn a_scene_has_a_history_and_rolls_back_by_compose() {
    let scene = tally_world("structure", 1_000_000);
    let first = std::fs::read_to_string(&scene).unwrap();
    let second = first.replace("100 / (at(count, 0) - 1000000)", "7 + at(count, 0)");
    assert_ne!(first, second);
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "document" }),
            json!({ "id": 2, "op": "compose", "text": second }),
            json!({ "id": 3, "op": "history" }),
        ],
    );
    let old = responses[0]["ok"]["version"].as_str().unwrap().to_owned();
    let now = responses[1]["ok"]["version"].as_str().unwrap().to_owned();
    let scenes: Vec<&Json> = responses[2]["ok"]["changes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "scene")
        .collect();
    assert_eq!(scenes.first().unwrap()["version"], old.as_str());
    assert_eq!(scenes.last().unwrap()["version"], now.as_str());
    assert_eq!(
        scenes.last().unwrap()["origin"],
        json!({ "protocol": "stdin" })
    );
    // The old structure read back and composed again: rolled back.
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "compose", "text": first }),
            json!({ "id": 2, "op": "compose", "text": second }),
            json!({ "id": 3, "op": "document", "version": old }),
        ],
    );
    let text = responses[2]["ok"]["text"].as_str().unwrap().to_owned();
    assert_eq!(text, first);
    let responses = talk(
        &scene,
        &[json!({ "id": 1, "op": "compose", "text": text, "expect": now })],
    );
    assert!(responses[0].get("error").is_none(), "{}", responses[0]);
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), first);
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// With `structure: "snapshot"` the scene the snapshot ran in comes back
/// before its state; when the world then cannot run, the scene that
/// stood is put back with the state.
#[test]
fn a_world_restored_with_its_structure_and_put_back_when_it_cannot_run() {
    let scene = tally_world("restructure", 3);
    let first = std::fs::read_to_string(&scene).unwrap();
    let relaxed = first.replace("- 3)", "- 1000000)");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "compose", "text": relaxed }),
            json!({ "id": 2, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 4, "op": "snapshot" }),
        ],
    );
    let relaxed_at_two = responses[3]["ok"].clone();
    assert_eq!(counted(&relaxed_at_two), 2);
    assert!(
        relaxed_at_two["scene"]["text"]
            .as_str()
            .unwrap()
            .contains("1000000")
    );
    // The scene on disk is strict again (the check fails at 3): restoring
    // the relaxed structure with the state runs, and the scene is the
    // snapshot's.
    std::fs::write(&scene, &first).unwrap();
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "restore", "snapshot": relaxed_at_two, "laws": "current", "structure": "snapshot" }),
            json!({ "id": 2, "op": "document" }),
            json!({ "id": 3, "op": "evaluate", "time": 0.2 }),
        ],
    );
    assert!(responses[0].get("error").is_none(), "{}", responses[0]);
    assert!(
        responses[1]["ok"]["text"]
            .as_str()
            .unwrap()
            .contains("1000000")
    );
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    // A snapshot of the strict scene at two, restored with its structure:
    // its trial counts to three and divides by zero, so the relaxed scene
    // that stood is put back, and the state with it.
    std::fs::write(&scene, &first).unwrap();
    let strict = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "evaluate", "time": 0.1 }),
            json!({ "id": 3, "op": "snapshot" }),
        ],
    );
    let strict_at_two = strict[2]["ok"].clone();
    std::fs::write(&scene, &relaxed).unwrap();
    let ten = orla_formats::base64::base64(&10u32.to_le_bytes());
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "write", "node": "tally", "data": ten }),
            json!({ "id": 3, "op": "restore", "snapshot": strict_at_two, "laws": "current", "structure": "snapshot" }),
            json!({ "id": 4, "op": "document" }),
            json!({ "id": 5, "op": "snapshot" }),
        ],
    );
    assert!(responses[2].get("error").is_some(), "{}", responses[2]);
    assert_eq!(
        responses[3]["ok"]["text"].as_str().unwrap(),
        relaxed,
        "the scene that stood"
    );
    assert_eq!(counted(&responses[4]["ok"]), 10, "the state that stood");
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// A restore with its structure that is refused while the document on
/// disk is an edit that does not load: the scene that ran is put back
/// from memory, not read from that text, and the history says a restore
/// was refused. (It used to leave the snapshot's scene running and the
/// history saying nothing had moved.)
#[test]
fn a_refused_restore_puts_back_the_scene_that_ran_whatever_is_on_disk() {
    use orla_runtime::Context;
    use orla_session::{Laws, Session, Structure};
    let scene = tally_world("put-back", 3);
    let strict = std::fs::read_to_string(&scene).unwrap();
    let relaxed = strict.replace("- 3)", "- 1000000)");
    let context = |time: f32| Context {
        time,
        ..Context::default()
    };
    // A snapshot of the strict scene at two: restored, its trial counts
    // to three and divides by zero.
    let mut session = Session::open(&scene, None, 1 << 20);
    session.sync();
    session.evaluate(&context(0.0)).unwrap();
    session.evaluate(&context(0.1)).unwrap();
    let strict_at_two = session.snapshot().unwrap();
    // The relaxed scene runs; then the person half-types an edit.
    std::fs::write(&scene, &relaxed).unwrap();
    session.sync();
    session.evaluate(&context(0.2)).unwrap();
    let half = format!("{relaxed}node oops: ");
    std::fs::write(&scene, &half).unwrap();
    let (revision, _) = session.position();
    let refused = session.restore(
        &strict_at_two,
        Laws::Current,
        Structure::Snapshot,
        &context(0.3),
    );
    assert!(refused.is_err(), "{refused:?}");
    // What runs is the relaxed scene that ran, as it ran.
    assert_eq!(
        session.snapshot().unwrap().scene.as_deref(),
        Some(relaxed.as_str()),
        "the scene that ran"
    );
    session
        .evaluate(&context(0.4))
        .expect("the scene that ran runs on");
    assert_eq!(
        std::fs::read_to_string(&scene).unwrap(),
        half,
        "the edit on disk stands"
    );
    let (after, _) = session.position();
    assert!(after >= revision);
    assert!(
        session
            .attempts_since(revision)
            .iter()
            .any(|attempt| format!("{attempt:?}").contains("snapshot")),
        "the refused restore is recorded"
    );
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// A refused restore moves nothing: not the revision, not a Kernel's
/// generation, not the scene on disk, and not an edit a person made there
/// that the session had not read yet. Only the attempt is recorded.
#[test]
fn a_refused_restore_moves_nothing() {
    let scene = tally_world("refused-whole", 1_000_000);
    let taken = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "snapshot" }),
        ],
    )[1]["ok"]
        .clone();
    let with = |edit: &dyn Fn(&mut Json)| {
        let mut snapshot = taken.clone();
        edit(&mut snapshot);
        snapshot
    };
    // Its laws differ and its states name a port that is no state: the
    // laws would be published, then the state refused.
    let not_state = with(&|s| {
        s["laws"][0]["source"] = json!(
            s["laws"][0]["source"]
                .as_str()
                .unwrap()
                .replace("+ 1;", "+ 2;")
        );
        s["states"][0]["node"] = json!("zero");
    });
    // Its scene defines the tally the library already publishes: the
    // scene reads, and its Kernel is refused.
    let clash = with(&|s| {
        let text = s["scene"]["text"].as_str().unwrap().to_owned()
            + "kernel t.tally@1 { requires { field.read; field.write; } input previous: Rows<Int>; output result: Rows<Int>; compute { return at(previous, 0) + 5; } }\n";
        s["scene"]["text"] = json!(text);
    });
    let on_disk = std::fs::read_to_string(&scene).unwrap();
    let edited = format!("{on_disk}// a person's edit, not read yet\n");
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "status" }),
            json!({ "id": 3, "op": "kernel", "kernel": "t.tally@1" }),
            json!({ "id": 4, "op": "restore", "snapshot": not_state, "laws": "snapshot" }),
            json!({ "id": 5, "op": "status" }),
            json!({ "id": 6, "op": "kernel", "kernel": "t.tally@1" }),
        ],
    );
    assert!(responses[3].get("error").is_some(), "{}", responses[3]);
    let (before, after) = (&responses[1]["ok"], &responses[4]["ok"]);
    assert_eq!(after["revision"], before["revision"], "no revision");
    assert_eq!(
        responses[5]["ok"]["generation"], responses[2]["ok"]["generation"],
        "no generation"
    );
    assert!(
        responses[5]["ok"]["source"]
            .as_str()
            .unwrap()
            .contains("+ 1;")
    );
    // The person's edit waits on disk while a structure restore is refused.
    std::fs::write(&scene, &edited).unwrap();
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "restore", "snapshot": clash, "laws": "current", "structure": "snapshot" }),
            json!({ "id": 2, "op": "kernel", "kernel": "t.tally@1" }),
            json!({ "id": 3, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 4, "op": "history" }),
        ],
    );
    assert!(responses[0].get("error").is_some(), "{}", responses[0]);
    assert_eq!(
        std::fs::read_to_string(&scene).unwrap(),
        edited,
        "the edit stands"
    );
    assert!(
        responses[1]["ok"]["source"]
            .as_str()
            .unwrap()
            .contains("+ 1;"),
        "{}",
        responses[1]
    );
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    let attempts = responses[3]["ok"]["attempts"].as_array().unwrap();
    assert!(
        attempts
            .iter()
            .any(|a| a["subject"].as_str().unwrap().contains("snapshot"))
    );
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// Where a change came from is where it came from: a scene edited on disk
/// and found by a request's sync is its document's, the scene a compose
/// wrote is the request's; and `document` by version reads scenes only.
#[test]
fn a_change_found_on_disk_is_its_documents_and_versions_name_scenes() {
    let scene = tally_world("provenance", 1_000_000);
    let first = std::fs::read_to_string(&scene).unwrap();
    let mut child = talk(
        &scene,
        &[json!({ "id": 1, "op": "kernel", "kernel": "t.tally@1" })],
    );
    let tally_digest = format!(
        "{:016x}",
        orla_session::history::digest(child.remove(0)["ok"]["source"].as_str().unwrap())
    );
    std::fs::write(&scene, format!("{first}// edited on disk\n")).unwrap();
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "status" }),
            json!({ "id": 2, "op": "document", "version": tally_digest }),
            json!({ "id": 3, "op": "compose", "text": first }),
            json!({ "id": 4, "op": "history" }),
        ],
    );
    assert!(
        responses[1].get("error").is_some(),
        "a Kernel's source is no scene: {}",
        responses[1]
    );
    let scenes: Vec<&Json> = responses[3]["ok"]["changes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["kind"] == "scene")
        .collect();
    assert!(
        scenes[0]["origin"]["document"].as_str().is_some(),
        "{:?}",
        scenes
    );
    assert_eq!(
        scenes.last().unwrap()["origin"],
        json!({ "protocol": "stdin" })
    );
    // Edited on disk while the session runs, and found by a sync a
    // protocol request made: still its document's.
    let runtime = orla_session::Session::runtime_for("cpu", Default::default()).unwrap();
    let mut session = orla_session::Session::open_with(&scene, None, runtime);
    session.sync();
    session.act_for(orla_session::Origin::Protocol("socket 1".into()));
    std::fs::write(&scene, format!("{first}// edited again\n")).unwrap();
    let file = std::fs::File::options().write(true).open(&scene).unwrap();
    file.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(2))
        .unwrap();
    session.sync();
    let found = session
        .changes_since(0)
        .into_iter()
        .rev()
        .find(|r| matches!(r.change, orla_session::Change::Scene { .. }));
    assert_eq!(
        found.unwrap().origin,
        orla_session::Origin::Document(scene.clone())
    );
    let _ = std::fs::remove_dir_all(scene.parent().unwrap());
}

/// `reset` with a `node` starts that one port again from its initial
/// source and leaves the rest of the world's state; history says which.
#[test]
fn a_reset_can_start_one_port_again_and_leave_the_rest() {
    let dir = std::env::temp_dir().join(format!("orla-protocol-reset-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let scene = dir.join("scene.osn");
    std::fs::write(
        &scene,
        "kernel t.zero@1 { requires { field.write; } output result: Rows<Float>; compute { return 0.0; } }
kernel t.count@1 { requires { field.read; field.write; } input previous: Rows<Float>; output result: Rows<Float>; compute { return sample(previous) + 1.0; } }
node seed: t.zero@1 { domain = 1; }
node a: t.count@1 { previous <~ a.result from seed.result; }
node b: t.count@1 { previous <~ b.result from seed.result; }
output a <- a.result;
output b <- b.result;
",
    )
    .unwrap();
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "advance", "frames": 3 }),
            json!({ "id": 2, "op": "reset", "node": "a" }),
            json!({ "id": 3, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 4, "op": "read", "node": "a" }),
            json!({ "id": 5, "op": "read", "node": "b" }),
            json!({ "id": 6, "op": "reset", "node": "seed" }),
            json!({ "id": 7, "op": "reset", "port": "result" }),
            json!({ "id": 8, "op": "reset", "node": "a", "port": 3 }),
            json!({ "id": 9, "op": "reset" }),
            json!({ "id": 10, "op": "history" }),
        ],
    );
    let ok = |i: usize| -> &Json {
        assert!(responses[i].get("error").is_none(), "{}", responses[i]);
        &responses[i]["ok"]
    };
    let failed = |i: usize| -> &str {
        responses[i]["error"]["message"]
            .as_str()
            .unwrap_or_else(|| panic!("{}", responses[i]))
    };
    assert_eq!(
        ok(1),
        &json!({ "reset": true, "node": "a", "port": "result" })
    );
    let value = |i: usize| {
        let bytes = orla_formats::base64::unbase64(ok(i)["data"].as_str().unwrap()).unwrap();
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    };
    assert_eq!((value(3), value(4)), (1.0, 4.0), "a again, b on");
    assert!(failed(5).contains("is not state"), "{}", failed(5));
    assert!(failed(6).contains("name the node"), "{}", failed(6));
    assert!(failed(7).contains("must be a string"), "{}", failed(7));
    assert_eq!(ok(8), &json!({ "reset": true }));
    let changes = ok(9)["changes"].as_array().unwrap();
    let resets: Vec<&Json> = changes.iter().filter(|c| c["kind"] == "reset").collect();
    assert_eq!(
        resets.len(),
        2,
        "the refused ones moved nothing: {changes:?}"
    );
    assert_eq!(resets[0]["node"], "a");
    assert_eq!(resets[0]["port"], "result");
    assert!(resets[1].get("node").is_none(), "{}", resets[1]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// `compose` with `edits` changes what they name and keeps the rest of
/// the text as written; an edit that cannot be made, or a result that
/// does not load, leaves the document as it was.
#[test]
fn compose_edits_the_document_where_it_changes() {
    let dir = scratch_in("-compose-edits");
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "document" }),
            // A member changed, a node added, the output moved to it.
            json!({ "id": 2, "op": "compose", "edits": [
                { "node": "life", "member": "brush", "text": "brush = 3.0;" },
                { "node": "dim", "text": "node dim: life.show@1 { cells <- life.result; }" },
                { "output": "result", "text": "output result <- dim.result;" },
            ] }),
            json!({ "id": 3, "op": "document" }),
            json!({ "id": 4, "op": "evaluate", "time": 0.0 }),
            // Removing what another node reads loads nothing.
            json!({ "id": 5, "op": "compose", "edits": [ { "node": "life", "text": null } ] }),
            // An edit that cannot be made writes nothing, the others with it.
            json!({ "id": 6, "op": "compose", "edits": [
                { "node": "life", "member": "brush", "text": "brush = 9.0;" },
                { "node": "nowhere", "text": null },
            ] }),
            json!({ "id": 7, "op": "compose", "edits": [ { "node": "life", "member": "brush" } ] }),
            json!({ "id": 8, "op": "compose", "edits": [ { "output": "result", "member": "x", "text": "x = 1;" } ] }),
            json!({ "id": 9, "op": "compose", "text": original, "edits": [] }),
            json!({ "id": 10, "op": "document" }),
        ],
    );
    let ok = |i: usize| -> &Json {
        assert!(responses[i].get("error").is_none(), "{}", responses[i]);
        &responses[i]["ok"]
    };
    let failed = |i: usize| -> &str {
        responses[i]["error"]["message"]
            .as_str()
            .unwrap_or_else(|| panic!("{}", responses[i]))
    };
    let edited = original
        .replace("brush = 2.0;", "brush = 3.0;")
        .replace(
            "node show: life.show@1 {\n    cells <- life.result;\n}\n",
            "node show: life.show@1 {\n    cells <- life.result;\n}\nnode dim: life.show@1 { cells <- life.result; }\n",
        )
        .replace("output result <- show.result;", "output result <- dim.result;");
    assert!(
        ok(1)["events"]
            .as_array()
            .unwrap()
            .iter()
            .any(|e| e["kind"] == "scene_loaded"),
        "{}",
        ok(1)
    );
    // Written as the edits say, the comment at the top as it was.
    assert_eq!(ok(2)["text"], edited);
    assert_eq!(ok(2)["version"], ok(1)["version"]);
    ok(3);
    assert_eq!(ok(4)["events"][0]["kind"], "scene_rejected", "{}", ok(4));
    assert!(
        failed(5).contains("edit 1: there is no node `nowhere` to remove"),
        "{}",
        failed(5)
    );
    assert!(
        failed(6).contains("`text` is what stands there"),
        "{}",
        failed(6)
    );
    assert!(
        failed(7).contains("only a node has members"),
        "{}",
        failed(7)
    );
    assert!(failed(8).contains("not both"), "{}", failed(8));
    assert_eq!(ok(9)["text"], edited, "nothing refused reached the disk");
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), edited);
    let _ = std::fs::remove_dir_all(&dir);
}

/// A node's own value is the document's: `param` reaches a node only
/// through a `param` the document declares, a value a node sets itself
/// changes by an edit of the document, and the document is where it
/// stays.
#[test]
fn a_nodes_own_value_is_the_documents() {
    let dir = scratch_in("-node-values");
    let scene = dir.join("scene.osn");
    let brush = |graph: &Json| {
        graph["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|n| n["id"] == "life")
            .map(|n| n["params"]["brush"].clone())
            .unwrap()
    };
    let responses = talk(
        &scene,
        &[
            // `brush` is life's own value, not a param of the scene.
            json!({ "id": 1, "op": "param", "name": "brush", "value": 5.0 }),
            json!({ "id": 2, "op": "graph" }),
            // An edit of the node changes it, in the document.
            json!({ "id": 3, "op": "compose", "edits": [
                { "node": "life", "member": "brush", "text": "brush = 5.0;" },
            ] }),
            json!({ "id": 4, "op": "graph" }),
            json!({ "id": 5, "op": "sync" }),
            json!({ "id": 6, "op": "graph" }),
            // Declared, a param reaches the node that names it, and the
            // document stays as written.
            json!({ "id": 7, "op": "compose", "edits": [
                { "param": "size", "text": "param size: Float = 2.0;" },
                { "node": "life", "member": "brush", "text": "brush = size;" },
            ] }),
            json!({ "id": 8, "op": "document" }),
            json!({ "id": 9, "op": "param", "name": "size", "value": 7.0 }),
            json!({ "id": 10, "op": "graph" }),
            json!({ "id": 11, "op": "document" }),
            json!({ "id": 12, "op": "param", "name": "sise", "value": 7.0 }),
        ],
    );
    let ok = |i: usize| -> &Json {
        assert!(responses[i].get("error").is_none(), "{}", responses[i]);
        &responses[i]["ok"]
    };
    let refused = responses[0]["error"]["message"].as_str().unwrap();
    assert!(
        refused.contains("`brush` is not a param the scene declares")
            && refused.contains("node `life` sets its own `brush`")
            && refused.contains("member `brush`"),
        "{refused}"
    );
    assert_eq!(
        brush(ok(1)),
        json!(2.0),
        "the refused param reached nothing"
    );
    ok(2);
    assert_eq!(brush(ok(3)), json!(5.0));
    assert!(ok(4)["events"].as_array().unwrap().is_empty(), "{}", ok(4));
    assert_eq!(brush(ok(5)), json!(5.0), "read again, the document says 5");
    ok(6);
    let version = ok(7)["version"].clone();
    assert!(ok(7)["text"].as_str().unwrap().contains("brush = size;"));
    assert_eq!(ok(8)["value"], json!(7.0));
    assert_eq!(brush(ok(9)), json!({ "param": "size" }));
    let size = ok(9)["params"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["name"] == "size")
        .unwrap()
        .clone();
    assert_eq!(
        (size["default"].clone(), size["value"].clone()),
        (json!(2.0), json!(7.0))
    );
    assert_eq!(
        ok(10)["version"],
        version,
        "a param is not written into the document"
    );
    let near = responses[11]["error"]["message"].as_str().unwrap();
    assert!(near.contains("did you mean `size`?"), "{near}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A scene update that cannot run whole does not replace the scene
/// running: a composed text whose own Kernel requires what the session
/// does not grant is refused before it reaches the disk.
#[test]
fn a_scene_update_that_cannot_run_leaves_the_scene_running() {
    let dir = scratch_in("-atomic-scene");
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let wanting = original.replace(
        "output result <- show.result;",
        "node t: kernel { requires { field.read; field.write; time.read; } input source: Image <- show.result; output result: Image; compute { let c = sample(source); return vec4(c.r * time(), c.g, c.b, 1.0); } }\noutput result <- t.result;",
    );
    assert_ne!(wanting, original);
    let responses = talk_with(
        &scene,
        &["--grant", "field.read,field.write,input.read"],
        &[
            json!({ "id": 1, "op": "evaluate", "time": 0.0 }),
            json!({ "id": 2, "op": "compose", "text": wanting }),
            json!({ "id": 3, "op": "evaluate", "time": 0.1 }),
        ],
    );
    let refused = &responses[1]["ok"];
    assert_eq!(refused["events"][0]["kind"], "scene_rejected", "{refused}");
    assert!(
        refused["events"][0]["message"]
            .as_str()
            .is_some_and(|m| m.contains("does not grant")),
        "{refused}"
    );
    assert!(responses[2].get("error").is_none(), "{}", responses[2]);
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), original);

    let _ = std::fs::remove_dir_all(&dir);
}

/// Where a served scene may read is the person's to say: the scene they
/// opened and what they allowed. A document an agent composes cannot
/// widen it by what it uses or places, and naming a file outside is
/// refused before anything of it is read.
#[test]
fn a_composed_scene_cannot_widen_what_it_may_read() {
    let dir = scratch_in("-held-reads");
    let outside = std::env::temp_dir().join(format!("orla-outside-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&outside);
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(outside.join("far.orla"), "kernel far.away@1 { requires { field.write; } output result: Image; compute { return vec4(1.0, 0.0, 0.0, 1.0); } }").unwrap();
    std::fs::write(outside.join("secret.txt"), "not for the agent").unwrap();
    let scene = dir.join("scene.osn");
    let original = std::fs::read_to_string(&scene).unwrap();
    let using = format!("use \"{}/\";\n{original}", outside.display());
    let placing = format!(
        "{original}\nnode far: graph \"{}\" {{ }}\n",
        outside.join("secret.txt").display()
    );
    let responses = talk(
        &scene,
        &[
            json!({ "id": 1, "op": "compose", "text": using }),
            json!({ "id": 2, "op": "compose", "text": placing }),
            json!({ "id": 3, "op": "kernels" }),
        ],
    );
    for response in &responses[..2] {
        let answer = &response["ok"];
        assert_eq!(answer["events"][0]["kind"], "scene_rejected", "{answer}");
        let message = answer["events"][0]["message"].as_str().unwrap_or_default();
        assert!(message.contains("is not read"), "{message}");
        assert!(
            !message.contains("not for the agent"),
            "nothing of the file is said: {message}"
        );
    }
    assert!(
        !responses[2].to_string().contains("far.away"),
        "{}",
        responses[2]
    );
    assert_eq!(std::fs::read_to_string(&scene).unwrap(), original);
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&outside);
}

/// A native operation is the Host's own: no request retires it, and no
/// Kernel takes its id.
#[test]
fn a_native_operation_is_neither_retired_nor_replaced() {
    let dir = scratch_in("-native-kept");
    let replacement = "kernel native.rasterize@1 { requires { field.read; field.write; } input source: Image; output result: Image; compute { return sample(source); } }";
    let responses = talk(
        &dir.join("scene.osn"),
        &[
            json!({ "id": 1, "op": "retire", "kernel": "native.rasterize@1" }),
            json!({ "id": 2, "op": "publish", "source": replacement, "expect": "any" }),
            json!({ "id": 3, "op": "publish", "source": replacement, "expect": "absent" }),
            json!({ "id": 4, "op": "kernel", "kernel": "native.rasterize@1" }),
        ],
    );
    let retired = responses[0]["error"]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(retired.contains("native operation"), "{}", responses[0]);
    for i in [1, 2] {
        assert!(responses[i].get("error").is_some(), "{}", responses[i]);
    }
    assert_eq!(responses[3]["ok"]["generation"], 0, "{}", responses[3]);
    let _ = std::fs::remove_dir_all(&dir);
}

/// The interpreter, slow to prepare, so a replacement prepared off the
/// frame is still being prepared when the request that asked for it
/// has been read.
struct Slow;

impl orla_runtime::Backend for Slow {
    fn name(&self) -> &'static str {
        "slow"
    }

    fn prepare(
        &self,
        module: &orla_runtime::Validated,
    ) -> Result<Box<dyn orla_runtime::Prepared>, String> {
        std::thread::sleep(std::time::Duration::from_millis(40));
        orla_runtime::Backend::prepare(&orla_cpu::CpuBackend, module)
    }

    fn execute(
        &self,
        prepared: &dyn orla_runtime::Prepared,
        module: &orla_runtime::Validated,
        invocation: &orla_runtime::Invocation,
        scope: &mut orla_runtime::Scope<'_>,
    ) -> Result<Vec<orla_runtime::Handle>, orla_runtime::ExecutionError> {
        orla_runtime::Backend::execute(&orla_cpu::CpuBackend, prepared, module, invocation, scope)
    }
}

/// A `compose` or a `publish` whose change is prepared off the frame is
/// answered once it stands, with what it came to, and nothing waits for
/// it meanwhile: another connection is answered at once, and only what
/// its own connection sent after it waits its turn. (A `compose` used to
/// hold the whole session until its Kernels were prepared, a window's
/// frames with it.)
#[test]
fn a_change_is_answered_once_it_stands_and_nothing_waits_for_it() {
    use orla_protocol::{Answers, Incoming, Reply};
    use std::sync::mpsc::{Receiver, channel};
    let dir = scratch_in("-later");
    let scene = dir.join("scene.osn");
    let text = |red: &str| {
        format!(
            "node wave: kernel k.wave@1 {{ requires {{ field.write; }} output result: Image; domain = vec2(4, 4); compute {{ return vec4({red}, 0.5, 0.25, 1.0); }} }}\noutput result <- wave.result;\n"
        )
    };
    std::fs::write(&scene, text("0.1")).unwrap();
    let runtime = orla_runtime::Runtime::with_backend(1 << 20, Box::new(Slow));
    let mut session = orla_session::Session::open_with(&scene, None, runtime);
    session.sync();
    session.set_background(true);
    let mut answers = Answers::new();
    let connection = |from: &str| {
        let (send, receive) = channel::<String>();
        (from.to_owned(), send, receive)
    };
    let (agent, to_agent, agent_hears) = connection("socket 1");
    let (other, to_other, other_hears) = connection("socket 2");
    let ask = |answers: &mut Answers,
               session: &mut orla_session::Session,
               from: &str,
               to: &std::sync::mpsc::Sender<String>,
               request: Json| {
        let incoming = Incoming {
            line: request.to_string(),
            reply: Reply::Connection(to.clone()),
            local: false,
            from: from.to_owned(),
        };
        answers.answer(session, incoming);
    };
    let heard = |receiver: &Receiver<String>| -> Option<Json> {
        receiver
            .try_recv()
            .ok()
            .map(|line| serde_json::from_str(&line).unwrap())
    };
    let wait = |answers: &mut Answers,
                session: &mut orla_session::Session,
                receiver: &Receiver<String>|
     -> Json {
        for _ in 0..10_000 {
            answers.poll(session);
            if let Some(answer) = heard(receiver) {
                return answer;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        panic!("never answered");
    };

    ask(
        &mut answers,
        &mut session,
        &agent,
        &to_agent,
        json!({ "id": 1, "op": "compose", "text": text("0.9") }),
    );
    ask(
        &mut answers,
        &mut session,
        &agent,
        &to_agent,
        json!({ "id": 2, "op": "status" }),
    );
    ask(
        &mut answers,
        &mut session,
        &other,
        &to_other,
        json!({ "id": 3, "op": "status" }),
    );
    assert!(answers.waiting(), "the compose waits for its Kernel");
    assert!(
        heard(&agent_hears).is_none(),
        "nor is what came after it answered"
    );
    let now = heard(&other_hears).expect("another connection is answered at once");
    assert_eq!(now["id"], 3);
    let composed = wait(&mut answers, &mut session, &agent_hears);
    assert_eq!(composed["id"], 1, "{composed}");
    let kinds: Vec<&str> = composed["ok"]["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|event| event["kind"].as_str())
        .collect();
    assert!(
        kinds.contains(&"published") && kinds.contains(&"scene_loaded"),
        "{kinds:?}"
    );
    assert_eq!(
        heard(&agent_hears).expect("then what came after it")["id"],
        2
    );

    // A publish, prepared off the frame, answers with its generation.
    let source = "kernel k.wave@1 { requires { field.write; } output result: Image; compute { return vec4(0.3, 0.5, 0.25, 1.0); } }";
    ask(
        &mut answers,
        &mut session,
        &agent,
        &to_agent,
        json!({ "id": 4, "op": "publish", "source": source, "expect": "any" }),
    );
    assert!(answers.waiting());
    ask(
        &mut answers,
        &mut session,
        &other,
        &to_other,
        json!({ "id": 5, "op": "status" }),
    );
    assert_eq!(heard(&other_hears).expect("answered at once")["id"], 5);
    let published = wait(&mut answers, &mut session, &agent_hears);
    assert_eq!(published["id"], 4, "{published}");
    assert!(
        published["ok"]["generation"].as_u64().unwrap() >= 2,
        "{published}"
    );
    assert!(!answers.waiting());
    let _ = std::fs::remove_dir_all(&dir);
}
