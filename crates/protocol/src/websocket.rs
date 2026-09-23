//! WebSocket framing for the protocol's socket, so a browser can be a
//! client: the handshake of RFC 6455 and its text, close, ping, and pong
//! frames, nothing more. A connection that opens with an HTTP request is
//! served this way; one that opens with a JSON line is served as lines.
//! The messages inside are the same JSON lines either way.

use std::io::{self, BufRead, Read, Write};

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

/// Reads the HTTP request the peer opened with. When it asks to upgrade to
/// a WebSocket, answers with the switch and returns `true`; otherwise
/// answers that an upgrade is required and returns `false`.
pub fn handshake(reader: &mut impl BufRead, writer: &mut impl Write) -> io::Result<bool> {
    let mut key = None;
    let mut upgrade = false;
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Ok(false);
        }
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((name, value)) = line.split_once(':') {
            let value = value.trim();
            match name.trim().to_ascii_lowercase().as_str() {
                "sec-websocket-key" => key = Some(value.to_owned()),
                "upgrade" if value.eq_ignore_ascii_case("websocket") => upgrade = true,
                _ => {}
            }
        }
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
        let mut payload = vec![0u8; length as usize];
        reader.read_exact(&mut payload)?;
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
        assert!(handshake(&mut reader, &mut reply).unwrap());
        let reply = String::from_utf8(reply).unwrap();
        assert!(reply.starts_with("HTTP/1.1 101"), "{reply}");
        assert!(
            reply.contains("Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=\r\n"),
            "{reply}"
        );
        let plain = b"GET / HTTP/1.1\r\nHost: x\r\n\r\n";
        let mut reader = &plain[..];
        let mut reply = Vec::new();
        assert!(!handshake(&mut reader, &mut reply).unwrap());
        assert!(
            String::from_utf8(reply)
                .unwrap()
                .starts_with("HTTP/1.1 426")
        );
    }
}
