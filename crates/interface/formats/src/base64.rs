//! Standard base64, for protocol and glTF bytes.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// Each byte's six bits, `SKIP` for padding and white space, `BAD` otherwise.
const SKIP: u8 = 0xfe;
const BAD: u8 = 0xff;
const VALUES: [u8; 256] = {
    let mut table = [BAD; 256];
    let mut index = 0;
    while index < 64 {
        table[ALPHABET[index] as usize] = index as u8;
        index += 1;
    }
    table[b'=' as usize] = SKIP;
    let spaces = [b' ', b'\t', b'\n', b'\r', 0x0c];
    let mut space = 0;
    while space < spaces.len() {
        table[spaces[space] as usize] = SKIP;
        space += 1;
    }
    table
};

/// Decodes standard base64, with or without padding; padding and white
/// space anywhere are passed over.
pub fn unbase64(text: &str) -> Result<Vec<u8>, String> {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 4 * 3 + 3);
    // Whole groups of four letters make three bytes each.
    let mut rest = 0;
    while rest + 4 <= bytes.len() {
        let group = &bytes[rest..rest + 4];
        let [a, b, c, d] = [
            VALUES[group[0] as usize],
            VALUES[group[1] as usize],
            VALUES[group[2] as usize],
            VALUES[group[3] as usize],
        ];
        if (a | b | c | d) >= 64 {
            break;
        }
        let word = (a as u32) << 18 | (b as u32) << 12 | (c as u32) << 6 | d as u32;
        out.extend_from_slice(&[(word >> 16) as u8, (word >> 8) as u8, word as u8]);
        rest += 4;
    }
    // The rest (padding, white space, a short end) a letter at a time.
    let mut acc: u32 = 0;
    let mut bits = 0;
    for &byte in &bytes[rest..] {
        let value = VALUES[byte as usize];
        if value == SKIP {
            continue;
        }
        if value == BAD {
            return Err(format!("`{}` is not base64", byte as char));
        }
        acc = (acc << 6) | value as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    Ok(out)
}

/// Standard base64 with padding.
pub fn base64(bytes: &[u8]) -> String {
    let mut out = Vec::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut groups = bytes.chunks_exact(3);
    for group in groups.by_ref() {
        let word = (group[0] as u32) << 16 | (group[1] as u32) << 8 | group[2] as u32;
        out.extend_from_slice(&[
            ALPHABET[(word >> 18) as usize & 63],
            ALPHABET[(word >> 12) as usize & 63],
            ALPHABET[(word >> 6) as usize & 63],
            ALPHABET[word as usize & 63],
        ]);
    }
    let tail = groups.remainder();
    if !tail.is_empty() {
        let word = (tail[0] as u32) << 16 | tail.get(1).map_or(0, |b| (*b as u32) << 8);
        out.push(ALPHABET[(word >> 18) as usize & 63]);
        out.push(ALPHABET[(word >> 12) as usize & 63]);
        out.push(if tail.len() == 2 {
            ALPHABET[(word >> 6) as usize & 63]
        } else {
            b'='
        });
        out.push(b'=');
    }
    String::from_utf8(out).expect("base64 letters are ASCII")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_decodes_what_it_encodes() {
        for sample in [
            &b""[..],
            b"f",
            b"fo",
            b"foo",
            b"foob",
            b"fooba",
            b"foobar",
            &[0, 255, 7, 128],
        ] {
            assert_eq!(unbase64(&base64(sample)).unwrap(), sample);
        }
        assert_eq!(unbase64("Zm9v").unwrap(), b"foo");
        assert!(unbase64("Zm9*").is_err());
        // Padding and white space anywhere are passed over, as before.
        assert_eq!(unbase64("Zm9v\nYmFy").unwrap(), b"foobar");
        assert_eq!(unbase64(" Zm 9v Ym Fy ").unwrap(), b"foobar");
        let long: Vec<u8> = (0..=255u8).cycle().take(1000).collect();
        assert_eq!(unbase64(&base64(&long)).unwrap(), long);
    }

    /// The letter-at-a-time decoder this one replaced: what it accepts and
    /// makes is the contract.
    fn reference(text: &str) -> Result<Vec<u8>, String> {
        let mut out = Vec::new();
        let (mut acc, mut bits) = (0u32, 0);
        for byte in text.bytes() {
            if byte == b'=' || byte.is_ascii_whitespace() {
                continue;
            }
            let value = ALPHABET
                .iter()
                .position(|c| *c == byte)
                .ok_or_else(|| format!("`{}` is not base64", byte as char))?;
            acc = (acc << 6) | value as u32;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push((acc >> bits) as u8);
                acc &= (1 << bits) - 1;
            }
        }
        Ok(out)
    }

    #[test]
    fn base64_decodes_as_the_reference_does() {
        let letters = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/= \n\t\r*-";
        let mut seed = 12345u32;
        for length in 0..200 {
            for _ in 0..8 {
                let text: String = (0..length)
                    .map(|_| {
                        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                        // Mostly letters, now and then padding, space or a bad byte.
                        let pick = (seed >> 8) as usize % 100;
                        letters[if pick < 92 {
                            pick % 64
                        } else {
                            64 + pick % (letters.len() - 64)
                        }] as char
                    })
                    .collect();
                assert_eq!(unbase64(&text), reference(&text), "{text:?}");
            }
        }
    }

    #[test]
    fn base64_matches_the_standard() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }
}
