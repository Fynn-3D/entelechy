//! WebSocket framing for the protocol's socket, so a browser can be a
//! client: the handshake of RFC 6455 and its text, close, ping, and pong
//! frames, nothing more. A connection that opens with an HTTP request is
//! served this way; one that opens with a JSON line is served as lines.
//! The messages inside are the same JSON lines either way.

use std::io::{self, BufRead, Read, Write};

/// The longest request a socket takes, in bytes: a line, or a
/// WebSocket message whole. What a peer says it will send is weighed
/// against this before anything is held for it, so no peer can make the
/// server take more than this for one request.
pub const MAX_MESSAGE: usize = 256 << 20;

/// The longest line of a handshake's header.
const MAX_HEADER_LINE: usize = 8 << 10;

/// Reads one line into `line`, the newline included, of at most `limit`
/// bytes; the bytes read, 0 at the end of the stream. A longer line is
/// refused before more than `limit` bytes of it are held.
pub(crate) fn read_line_within(
    reader: &mut impl BufRead,
    limit: usize,
    line: &mut Vec<u8>,
) -> io::Result<usize> {
    line.clear();
    let read = (&mut *reader)
        .take(limit as u64 + 1)
        .read_until(b'\n', line)?;
    if line.len() > limit {
        return Err(too_long());
    }
    Ok(read)
}

/// The refusal of a request past [`MAX_MESSAGE`].
pub(crate) fn too_long() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidData,
        format!("a request is at most {MAX_MESSAGE} bytes"),
    )
}

/// A message read from the socket.
#[derive(Debug, PartialEq, Eq)]
pub enum Message {
    /// A text (or binary) message: one request line.
    Text(String),
    /// A ping, to answer with a pong carrying the same payload.
    Ping(Vec<u8>),
    /// The peer is closing.
    Close,
}

/// The web pages whose scripts a socket answers. A browser says which
/// page opened a WebSocket in its `Origin`, and any page on the web can
/// open one to this machine: without a check, a site the person happens
/// to visit could compose their scene and read what it holds. Admitted
/// are a client that is not a browser (no `Origin`), a page opened from a
/// file (`null`, which is how `bridges/web` runs), a page served from
/// this machine (`localhost`, `127.0.0.1`, `[::1]`, any port), and the
/// origins the person running the server named (`--allow-origin`).
#[derive(Clone, Debug, Default)]
pub struct Origins {
    named: Vec<String>,
}

impl Origins {
    /// The local origins and these, each written as a browser sends it:
    /// `https://example.com`, `http://host:8080`.
    pub fn with(named: Vec<String>) -> Self {
        let named = named
            .into_iter()
            .map(|origin| origin.trim_end_matches('/').to_ascii_lowercase())
            .collect();
        Self { named }
    }

    /// Whether a page of `origin` may open the socket.
    pub fn admits(&self, origin: Option<&str>) -> bool {
        let Some(origin) = origin else {
            return true;
        };
        let origin = origin.trim().trim_end_matches('/').to_ascii_lowercase();
        if origin == "null" || self.named.contains(&origin) {
            return true;
        }
        let Some(rest) = origin
            .strip_prefix("http://")
            .or_else(|| origin.strip_prefix("https://"))
        else {
            return false;
        };
        // `host` or `host:port`, the port all digits: nothing may follow
        // the host that would make it another one.
        let (host, port) = match rest.find(']') {
            Some(end) if rest.starts_with('[') => (&rest[..=end], &rest[end + 1..]),
            _ => rest.split_at(rest.find(':').unwrap_or(rest.len())),
        };
        let port_ok = port.is_empty()
            || port.strip_prefix(':').is_some_and(|digits| {
                !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
            });
        port_ok && matches!(host, "localhost" | "127.0.0.1" | "[::1]")
    }
}

/// Reads the HTTP request the peer opened with. When it asks to upgrade to
/// a WebSocket from an origin `origins` admits, answers with the switch
/// and returns `true`; a page it does not admit is refused, and a request
/// that is no upgrade is told one is required, both returning `false`.
pub fn handshake(
    reader: &mut impl BufRead,
    writer: &mut impl Write,
    origins: &Origins,
) -> io::Result<bool> {
    let mut key = None;
    let mut upgrade = false;
    let mut origin = None;
    let mut bytes = Vec::new();
    loop {
        if read_line_within(reader, MAX_HEADER_LINE, &mut bytes)? == 0 {
            return Ok(false);
        }
        let line = String::from_utf8_lossy(&bytes);
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "sec-websocket-key" => key = Some(value.to_owned()),
                "upgrade" if value.eq_ignore_ascii_case("websocket") => upgrade = true,
                "origin" => origin = Some(value.to_owned()),
                _ => {}
            }
        }
    }
    if !origins.admits(origin.as_deref()) {
        let body = format!(
            "A page of {} may not open this session; the person running it may allow it with --allow-origin.\n",
            origin.unwrap_or_default()
        );
        write!(
            writer,
            "HTTP/1.1 403 Forbidden\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        )?;
        writer.flush()?;
        return Ok(false);
    }
    match key {
        Some(key) if upgrade => {
            write!(
                writer,
                "HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: {}\r\n\r\n",
                accept_key(&key)
            )?;
            writer.flush()?;
            Ok(true)
        }
        _ => {
            let body = "This is Orla's creation protocol: JSON lines on a socket, or the same lines as WebSocket text frames. See docs/PROTOCOL.md.\n";
            write!(
                writer,
                "HTTP/1.1 426 Upgrade Required\r\nUpgrade: websocket\r\nConnection: close\r\nContent-Type: text/plain\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            )?;
            writer.flush()?;
            Ok(false)
        }
    }
}

/// The `Sec-WebSocket-Accept` value for a client's key.
pub fn accept_key(key: &str) -> String {
    let mut material = key.trim().as_bytes().to_vec();
    material.extend_from_slice(b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    orla_formats::base64::base64(&sha1(&material))
}

/// Reads one message, joining fragments; `Ok(None)` when the peer closed
/// the stream. Pongs are read and dropped.
pub fn read_message(reader: &mut impl Read) -> io::Result<Option<Message>> {
    let mut text: Vec<u8> = Vec::new();
    loop {
        let mut head = [0u8; 2];
        match reader.read_exact(&mut head) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(error) => return Err(error),
        }
        let fin = head[0] & 0x80 != 0;
        let opcode = head[0] & 0x0f;
        let masked = head[1] & 0x80 != 0;
        let mut length = (head[1] & 0x7f) as u64;
        if length == 126 {
            let mut ext = [0u8; 2];
            reader.read_exact(&mut ext)?;
            length = u16::from_be_bytes(ext) as u64;
        } else if length == 127 {
            let mut ext = [0u8; 8];
            reader.read_exact(&mut ext)?;
            length = u64::from_be_bytes(ext);
        }
        let mut mask = [0u8; 4];
        if masked {
            reader.read_exact(&mut mask)?;
        }
        // The length is the peer's word: weighed first, and what is held
        // grows with the bytes that do arrive.
        if length > (MAX_MESSAGE - text.len()) as u64 {
            return Err(too_long());
        }
        let mut payload = Vec::new();
        reader.by_ref().take(length).read_to_end(&mut payload)?;
        if payload.len() as u64 != length {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        if masked {
            for (index, byte) in payload.iter_mut().enumerate() {
                *byte ^= mask[index % 4];
            }
        }
        match opcode {
            0x8 => return Ok(Some(Message::Close)),
            0x9 => return Ok(Some(Message::Ping(payload))),
            0xA => continue,
            0x0..=0x2 => {
                text.extend_from_slice(&payload);
                if fin {
                    let text = String::from_utf8(text).map_err(|_| {
                        io::Error::new(io::ErrorKind::InvalidData, "message is not UTF-8")
                    })?;
                    return Ok(Some(Message::Text(text)));
                }
            }
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown frame opcode {other}"),
                ));
            }
        }
    }
}

/// Writes one unmasked frame, as a server does.
fn write_frame(writer: &mut impl Write, opcode: u8, payload: &[u8]) -> io::Result<()> {
    let mut head = vec![0x80 | opcode];
    let length = payload.len();
    if length < 126 {
        head.push(length as u8);
    } else if length <= u16::MAX as usize {
        head.push(126);
        head.extend_from_slice(&(length as u16).to_be_bytes());
    } else {
        head.push(127);
        head.extend_from_slice(&(length as u64).to_be_bytes());
    }
    writer.write_all(&head)?;
    writer.write_all(payload)?;
    writer.flush()
}

/// Writes a text message.
pub fn write_text(writer: &mut impl Write, text: &str) -> io::Result<()> {
    write_frame(writer, 0x1, text.as_bytes())
}

/// Answers a ping.
pub fn write_pong(writer: &mut impl Write, payload: &[u8]) -> io::Result<()> {
    write_frame(writer, 0xA, payload)
}

/// Says goodbye: a close frame with the normal status.
pub fn write_close(writer: &mut impl Write) -> io::Result<()> {
    write_frame(writer, 0x8, &1000u16.to_be_bytes())
}

/// SHA-1, needed for the handshake alone.
fn sha1(data: &[u8]) -> [u8; 20] {
    let mut h: [u32; 5] = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0];
    let mut message = data.to_vec();
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&((data.len() as u64) * 8).to_be_bytes());
    for block in message.chunks_exact(64) {
        let mut w = [0u32; 80];
        for (i, word) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..80 {
            w[i] = (w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16]).rotate_left(1);
        }
        let [mut a, mut b, mut c, mut d, mut e] = h;
        for (i, word) in w.iter().enumerate() {
            let (f, k) = match i {
                0..=19 => ((b & c) | (!b & d), 0x5A827999),
                20..=39 => (b ^ c ^ d, 0x6ED9EBA1),
                40..=59 => ((b & c) | (b & d) | (c & d), 0x8F1BBCDC),
                _ => (b ^ c ^ d, 0xCA62C1D6),
            };
            let temp = a
                .rotate_left(5)
                .wrapping_add(f)
                .wrapping_add(e)
                .wrapping_add(k)
                .wrapping_add(*word);
            e = d;
            d = c;
            c = b.rotate_left(30);
            b = a;
            a = temp;
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
    }
    let mut out = [0u8; 20];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn sha1_matches_the_known_vectors() {
        assert_eq!(hex(&sha1(b"")), "da39a3ee5e6b4b0d3255bfef95601890afd80709");
        assert_eq!(
            hex(&sha1(b"abc")),
            "a9993e364706816aba3e25717850c26c9cd0d89d"
        );
        assert_eq!(
            hex(&sha1(
                b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
            )),
            "84983e441c3bd26ebaae4aa1f95129e5e54670f1"
        );
    }

    /// A peer's word for a length is weighed before anything is held:
    /// a frame that says it carries 2^63 bytes is refused, not allocated.
    #[test]
    fn a_frame_longer_than_a_request_may_be_is_refused_before_it_is_held() {
        let mut frame = vec![0x81u8, 0x80 | 127];
        frame.extend_from_slice(&(u64::MAX >> 1).to_be_bytes());
        frame.extend_from_slice(&[1, 2, 3, 4]);
        let error = read_message(&mut io::Cursor::new(frame)).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("at most"), "{error}");
        // Fragments count together.
        let mut first = vec![0x01u8, 126];
        first.extend_from_slice(&60000u16.to_be_bytes());
        first.extend(std::iter::repeat_n(b'x', 60000));
        let mut stream = Vec::new();
        for _ in 0..(MAX_MESSAGE / 60000 + 1) {
            stream.extend_from_slice(&first);
        }
        let error = read_message(&mut io::Cursor::new(stream)).unwrap_err();
        assert!(error.to_string().contains("at most"), "{error}");
    }

    #[test]
    fn a_line_longer_than_its_limit_is_refused() {
        let mut line = Vec::new();
        let mut reader = io::Cursor::new(b"abc\nabcdefgh\n".to_vec());
        assert_eq!(read_line_within(&mut reader, 4, &mut line).unwrap(), 4);
        assert_eq!(line, b"abc\n");
        let error = read_line_within(&mut reader, 4, &mut line).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(line.len() <= 5, "no more than the limit is held");
    }

    #[test]
    fn the_accept_key_follows_the_rfc_example() {
        assert_eq!(
            accept_key("dGhlIHNhbXBsZSBub25jZQ=="),
            "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        );
    }

    #[test]
    fn masked_client_frames_read_back_and_server_frames_write() {
        // A masked text frame, as a browser sends it, in two fragments.
        let mask = [0x37, 0xfa, 0x21, 0x3d];
        let frame = |first: bool, fin: bool, text: &[u8]| {
            let mut out = vec![
                if fin { 0x80 } else { 0x00 } | if first { 0x1 } else { 0x0 },
                0x80 | text.len() as u8,
            ];
            out.extend_from_slice(&mask);
            out.extend(text.iter().enumerate().map(|(i, b)| b ^ mask[i % 4]));
            out
        };
        let mut bytes = frame(true, false, b"{\"op\":");
        bytes.extend(frame(false, true, b"\"status\"}"));
        bytes.extend([0x89, 0x80, 1, 2, 3, 4]); // a ping with no payload
        bytes.extend([0x88, 0x82, 0, 0, 0, 0, 0x03, 0xe8]); // close 1000
        let mut reader = &bytes[..];
        assert_eq!(
            read_message(&mut reader).unwrap(),
            Some(Message::Text("{\"op\":\"status\"}".into()))
        );
        assert_eq!(
            read_message(&mut reader).unwrap(),
            Some(Message::Ping(vec![]))
        );
        assert_eq!(read_message(&mut reader).unwrap(), Some(Message::Close));
        assert_eq!(read_message(&mut reader).unwrap(), None);

        let mut out = Vec::new();
        write_text(&mut out, "hi").unwrap();
        assert_eq!(out, [0x81, 2, b'h', b'i']);
        let long = "x".repeat(300);
        let mut out = Vec::new();
        write_text(&mut out, &long).unwrap();
        assert_eq!(&out[..4], [0x81, 126, 0x01, 0x2c]);
        assert_eq!(out.len(), 4 + 300);
    }

    #[test]
    fn the_handshake_switches_or_asks_for_an_upgrade() {
        let request = b"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
        let mut reader = &request[..];
        let mut reply = Vec::new();
        assert!(handshake(&mut reader, &mut reply, &Origins::default()).unwrap());
        let reply = String::from_utf8(reply).unwrap();
        assert!(reply.starts_with("HTTP/1.1 101"), "{reply}");
        assert!(
            reply.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n"),
            "{reply}"
        );
        let plain = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut reader = &plain[..];
        let mut reply = Vec::new();
        assert!(!handshake(&mut reader, &mut reply, &Origins::default()).unwrap());
        assert!(
            String::from_utf8(reply)
                .unwrap()
                .starts_with("HTTP/1.1 426")
        );
    }

    /// A page on the web may not open the socket; a page of this machine,
    /// one opened from a file, a client that is no browser, and an origin
    /// the person named may.
    #[test]
    fn a_page_elsewhere_is_refused_and_a_local_one_admitted() {
        let origins = Origins::with(vec!["https://studio.example".into()]);
        for admitted in [
            None,
            Some("null"),
            Some("http://localhost:8000"),
            Some("http://127.0.0.1"),
            Some("https://[::1]:9000"),
            Some("https://studio.example/"),
        ] {
            assert!(origins.admits(admitted), "{admitted:?}");
        }
        for refused in [
            "https://evil.example",
            "http://localhost.evil.example",
            "http://127.0.0.1.evil.example:80",
            "http://[::1].evil.example",
            "https://studio.example.evil",
            "file:///etc",
        ] {
            assert!(!origins.admits(Some(refused)), "{refused}");
        }
        let request = b"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nOrigin: https://evil.example\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n";
        let mut reader = &request[..];
        let mut reply = Vec::new();
        assert!(!handshake(&mut reader, &mut reply, &origins).unwrap());
        assert!(
            String::from_utf8(reply)
                .unwrap()
                .starts_with("HTTP/1.1 403")
        );
    }
}
