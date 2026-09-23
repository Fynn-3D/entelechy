//! How requests arrive and answers leave: standard input and output, a
//! socket of JSON lines, and a WebSocket carrying the same lines. The
//! transport is a detail; every request is answered by [`crate::handle`].

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

/// One request line and where its answer goes: the process's standard
/// output, or the socket it came in on. The transports below feed these
/// to whoever serves the session, a loop or a window.
pub struct Incoming {
    /// The request line.
    pub line: String,
    /// Where the response line goes.
    pub reply: Reply,
    /// Whether it came on standard input, where `quit` ends the process;
    /// on a socket, `quit` ends that connection.
    pub local: bool,
}

/// Where an answer goes.
#[derive(Clone)]
pub enum Reply {
    /// A connection's own writer, which outlives the answer.
    Connection(std::sync::mpsc::Sender<String>),
    /// The process's standard output, written where the answer is made.
    /// A `quit` on standard input ends the process, and a writer of its
    /// own might not have written the answer to it by then: this is why
    /// the last answer used to go missing about once in a hundred runs.
    Stdout,
}

impl Reply {
    /// Sends one answer line.
    pub fn send(&self, line: String) {
        match self {
            Self::Connection(sender) => {
                let _ = sender.send(line);
            }
            Self::Stdout => {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                let _ = writeln!(out, "{line}");
                let _ = out.flush();
            }
        }
    }
}

/// Feeds standard input's lines to `sink`; the answers are written on
/// standard output where they are made.
pub fn stdin_into(sink: std::sync::mpsc::Sender<Incoming>) {
    std::thread::spawn(move || {
        let stdin = std::io::stdin();
        for line in stdin.lock().lines().map_while(Result::ok) {
            if line.trim().is_empty() {
                continue;
            }
            let incoming = Incoming {
                line,
                reply: Reply::Stdout,
                local: true,
            };
            if sink.send(incoming).is_err() {
                break;
            }
        }
    });
}

/// Listens on `address` (`127.0.0.1:7788`, or port 0 for any) and feeds
/// every connection's lines to `sink`; each connection's answers go back
/// on its socket, one line each, from a thread of its own. Returns the
/// address bound. The protocol is the same on a socket as on standard
/// input, so a tool in another program, Blender say, talks to a running
/// window. A connection that opens with an HTTP request is a browser:
/// it is upgraded to a WebSocket and the same lines travel as text
/// frames ([`crate::websocket`]).
pub fn listen(
    address: &str,
    sink: std::sync::mpsc::Sender<Incoming>,
) -> std::io::Result<std::net::SocketAddr> {
    let listener = std::net::TcpListener::bind(address)?;
    let bound = listener.local_addr()?;
    std::thread::spawn(move || {
        for stream in listener.incoming().map_while(Result::ok) {
            let sink = sink.clone();
            std::thread::spawn(move || {
                let mut opening = [0u8; 4];
                let websocket = matches!(stream.peek(&mut opening), Ok(4) if &opening == b"GET ");
                if websocket {
                    serve_websocket(stream, sink);
                } else {
                    serve_lines(stream, sink);
                }
            });
        }
    });
    Ok(bound)
}

/// One connection of JSON lines.
fn serve_lines(stream: std::net::TcpStream, sink: std::sync::mpsc::Sender<Incoming>) {
    let Ok(mut writer) = stream.try_clone() else {
        return;
    };
    let (reply, answers) = std::sync::mpsc::channel::<String>();
    std::thread::spawn(move || {
        for line in answers {
            // An empty line is the sign to close: sent after a quit's
            // answer, it ends the connection from this side.
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
    let reader = std::io::BufReader::new(stream);
    for line in reader.lines().map_while(Result::ok) {
        if line.trim().is_empty() {
            continue;
        }
        let incoming = Incoming {
            line,
            reply: Reply::Connection(reply.clone()),
            local: false,
        };
        if sink.send(incoming).is_err() {
            break;
        }
    }
}

/// One connection of WebSocket frames: the handshake, then a text frame
/// per request line and one per answer; pings are answered, a close
/// frame or a quit ends it.
fn serve_websocket(stream: std::net::TcpStream, sink: std::sync::mpsc::Sender<Incoming>) {
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
        websocket::handshake(&mut reader, &mut *guard)
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
            Ok(Some(Message::Close)) | Ok(None) | Err(_) => {
                let _ = reply.send(String::new());
                break;
            }
        }
    }
}
