//! How requests arrive and answers leave: standard input and output, a
//! socket of JSON lines, or a WebSocket carrying the same lines. Every
//! request is answered by [`crate::handle`].

use std::io::{BufRead, Write};

use orla_session::Session;

use crate::respond_line;

/// Serves requests from `input` until EOF or `quit`, writing responses to
/// `output`.
pub fn serve(
    session: &mut Session,
    input: impl BufRead,
    mut output: impl Write,
) -> std::io::Result<()> {
    for line in input.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let (response, quit) = respond_line(session, &line);
        writeln!(output, "{response}")?;
        output.flush()?;
        if quit {
            break;
        }
    }
    Ok(())
}

/// One request line and where its answer goes. Transports feed these to
/// whatever serves the session, a loop or a window.
pub struct Incoming {
    /// The request line.
    pub line: String,
    /// Where the response line goes.
    pub reply: Reply,
    /// Whether it came on standard input, where `quit` ends the process;
    /// on a socket, `quit` ends the connection.
    pub local: bool,
    /// The channel it came on (`stdin`, `socket 3`, `websocket 4`, numbered
    /// in opening order), recorded as the origin of its changes.
    pub from: String,
}

/// Where an answer goes.
#[derive(Clone)]
pub enum Reply {
    /// A connection's own writer, which outlives the answer.
    Connection(std::sync::mpsc::Sender<String>),
    /// The process's standard output, written synchronously where the
    /// answer is made, so a `quit`'s answer is out before the process
    /// ends.
    Stdout,
    /// Standard output as an MCP server's tool result, under the call's
    /// JSON-RPC id ([`crate::mcp::result`]).
    Mcp(serde_json::Value),
}

/// How requests arrive on standard input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Framing {
    /// The protocol's own lines.
    Lines,
    /// MCP: JSON-RPC 2.0, each operation but `quit` a tool ([`crate::mcp`]).
    Mcp,
}

impl Reply {
    /// Sends one answer line.
    pub fn send(&self, line: String) {
        match self {
            Self::Connection(sender) => {
                let _ = sender.send(line);
            }
            Self::Stdout => write_stdout(&line),
            Self::Mcp(id) => {
                // The empty line that closes a connection has no result.
                if !line.is_empty() {
                    write_stdout(&crate::mcp::result(id, &line));
                }
            }
        }
    }
}

/// One line to standard output, flushed.
fn write_stdout(line: &str) {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let _ = writeln!(out, "{line}");
    let _ = out.flush();
}

/// Feeds standard input's requests to `sink` from its own thread. Where no
/// thread starts (WebAssembly) it returns the error, and the caller reads
/// [`stdin_lines`] itself, answering each before the next.
pub fn stdin_into(
    sink: std::sync::mpsc::Sender<Incoming>,
    framing: Framing,
) -> std::io::Result<()> {
    std::thread::Builder::new()
        .name("orla stdin".into())
        .spawn(move || {
            for incoming in stdin_lines(framing) {
                if sink.send(incoming).is_err() {
                    break;
                }
            }
        })
        .map(drop)
}

/// Standard input's requests, as they arrive. Under MCP, what needs no
/// session (`initialize`, `tools/list`, `ping`) is answered here, and a
/// tool call arrives as its operation's request.
pub fn stdin_lines(framing: Framing) -> impl Iterator<Item = Incoming> {
    std::io::stdin()
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty())
        .filter_map(move |line| match framing {
            Framing::Lines => Some(Incoming {
                line,
                reply: Reply::Stdout,
                local: true,
                from: "stdin".into(),
            }),
            Framing::Mcp => match crate::mcp::step(&line) {
                crate::mcp::Step::Answer(answer) => {
                    write_stdout(&answer);
                    None
                }
                crate::mcp::Step::Quiet => None,
                crate::mcp::Step::Call { id, request } => Some(Incoming {
                    line: request.to_string(),
                    reply: Reply::Mcp(id),
                    local: true,
                    from: "mcp".into(),
                }),
            },
        })
}

/// Listens on `address` (`127.0.0.1:7788`, or port 0 for any), feeds every
/// connection's lines to `sink`, and returns the bound address. Each
/// connection's answers go back on its socket from its own thread. A
/// connection opening with `GET` is upgraded to a WebSocket carrying the
/// same lines as text frames ([`crate::websocket`]), admitted by origin
/// ([`crate::websocket::Origins`]).
///
/// A connection opening with any other HTTP request is refused and
/// closed, so a web page's request body is never read as request lines.
pub fn listen(
    address: &str,
    sink: std::sync::mpsc::Sender<Incoming>,
    origins: crate::websocket::Origins,
) -> std::io::Result<std::net::SocketAddr> {
    let listener = std::net::TcpListener::bind(address)?;
    let bound = listener.local_addr()?;
    std::thread::spawn(move || {
        for (number, stream) in listener.incoming().map_while(Result::ok).enumerate() {
            let sink = sink.clone();
            let origins = origins.clone();
            let number = number + 1;
            std::thread::spawn(move || {
                let mut opening = [0u8; 8];
                let peeked = stream.peek(&mut opening).unwrap_or(0);
                let opening = &opening[..peeked];
                if opening.starts_with(b"GET ") {
                    serve_websocket(stream, sink, &origins, format!("websocket {number}"));
                } else if is_http(opening) {
                    refuse_http(stream);
                } else {
                    serve_lines(stream, sink, format!("socket {number}"));
                }
            });
        }
    });
    Ok(bound)
}

/// Whether a connection opens as an HTTP request: a capitalized method
/// word and a space, which a JSON request line never starts with.
fn is_http(opening: &[u8]) -> bool {
    let word = opening
        .iter()
        .take_while(|b| b.is_ascii_uppercase())
        .count();
    word >= 3 && opening.get(word) == Some(&b' ')
}

/// Answers a non-upgrade HTTP request with 400 and closes.
fn refuse_http(mut stream: std::net::TcpStream) {
    let body = "Orla's creation protocol takes JSON lines, or a WebSocket; see docs/PROTOCOL.md.\n";
    let _ = write!(
        stream,
        "HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let _ = stream.flush();
    let _ = stream.shutdown(std::net::Shutdown::Both);
}

/// Tells a connection why what it sent is refused, then closes it.
fn refuse(reply: &std::sync::mpsc::Sender<String>, error: &std::io::Error) {
    let refusal = crate::envelope(
        &serde_json::Value::Null,
        Err(crate::wire::failure(error.to_string())),
    );
    let _ = reply.send(refusal);
    let _ = reply.send(String::new());
}

/// One connection of JSON lines.
fn serve_lines(stream: std::net::TcpStream, sink: std::sync::mpsc::Sender<Incoming>, from: String) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in answers {
            // An empty line, sent after a quit's answer, closes the
            // connection.
            if line.is_empty()
                || writeln!(writer, "{line}")
                    .and_then(|_| writer.flush())
                    .is_err()
            {
                break;
            }
        }
        let _ = writer.shutdown(std::net::Shutdown::Both);
    });
    let mut reader = std::io::BufReader::new(stream);
    let mut bytes = Vec::new();
    loop {
        let line = match crate::websocket::read_line_within(
            &mut reader,
            crate::websocket::MAX_MESSAGE,
            &mut bytes,
        ) {
            Ok(0) => break,
            Ok(_) => match std::str::from_utf8(&bytes) {
                Ok(line) => line.trim_end_matches(['\n', '\r']).to_owned(),
                Err(_) => break,
            },
            Err(error) => {
                refuse(&reply, &error);
                break;
            }
        };
        if line.trim().is_empty() {
            continue;
        }
        let incoming = Incoming {
            line,
            reply: Reply::Connection(reply.clone()),
            local: false,
            from: from.clone(),
        };
        if sink.send(incoming).is_err() {
            break;
        }
    }
}

/// One WebSocket connection: the handshake, then a text frame per request
/// and per answer; pings are answered; a close frame or a quit ends it.
fn serve_websocket(
    stream: std::net::TcpStream,
    sink: std::sync::mpsc::Sender<Incoming>,
    origins: &crate::websocket::Origins,
    from: String,
) {
    use crate::websocket::{self, Message};
    let Ok(writer) = stream.try_clone() else {
        return;
    };
    let writer = std::sync::Arc::new(std::sync::Mutex::new(writer));
    let mut reader = std::io::BufReader::new(stream);
    let opened = {
        let mut guard = writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        websocket::handshake(&mut reader, &mut *guard, origins)
    };
    if !matches!(opened, Ok(true)) {
        let guard = writer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let _ = guard.shutdown(std::net::Shutdown::Both);
        return;
    }
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    let framer = writer.clone();
    std::thread::spawn(move || {
        for line in answers {
            let mut guard = framer
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let written = if line.is_empty() {
                websocket::write_close(&mut *guard).and(Err(std::io::Error::other("closed")))
            } else {
                websocket::write_text(&mut *guard, &line)
            };
            if written.is_err() {
                let _ = guard.shutdown(std::net::Shutdown::Both);
                break;
            }
        }
    });
    loop {
        match websocket::read_message(&mut reader) {
            Ok(Some(Message::Text(line))) => {
                if line.trim().is_empty() {
                    continue;
                }
                let incoming = Incoming {
                    line,
                    reply: Reply::Connection(reply.clone()),
                    local: false,
                    from: from.clone(),
                };
                if sink.send(incoming).is_err() {
                    break;
                }
            }
            Ok(Some(Message::Ping(payload))) => {
                let mut guard = writer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if websocket::write_pong(&mut *guard, &payload).is_err() {
                    break;
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::InvalidData => {
                refuse(&reply, &error);
                break;
            }
            Ok(Some(Message::Close)) | Ok(None) | Err(_) => {
                let _ = reply.send(String::new());
                break;
            }
        }
    }
}
