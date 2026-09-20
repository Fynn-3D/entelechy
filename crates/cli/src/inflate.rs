//! DEFLATE (RFC 1951) and zlib (RFC 1950) decoding, and a stored-block
//! zlib encoder, for reading and writing PNG files with no dependency.
//! The decoder is plain: canonical Huffman tables read one bit at a time,
//! which is fast enough for assets and simple enough to be obviously
//! right.

use std::fmt;

/// Why bytes are not a zlib stream.
#[derive(Debug)]
pub enum InflateError {
    /// The stream ended early or is malformed.
    Format(String),
    /// The Adler-32 trailer does not match the output.
    Checksum,
}

impl fmt::Display for InflateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Format(message) => write!(f, "bad deflate stream: {message}"),
            Self::Checksum => f.write_str("zlib checksum mismatch"),
        }
    }
}

impl std::error::Error for InflateError {}

struct Bits<'a> {
    bytes: &'a [u8],
    position: usize,
    acc: u32,
    count: u32,
}

impl Bits<'_> {
    fn need(&mut self, n: u32) -> Result<(), InflateError> {
        while self.count < n {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| InflateError::Format("stream ends early".into()))?;
            self.position += 1;
            self.acc |= (byte as u32) << self.count;
            self.count += 8;
        }
        Ok(())
    }

    fn bits(&mut self, n: u32) -> Result<u32, InflateError> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let value = self.acc & ((1u32 << n) - 1);
        self.acc >>= n;
        self.count -= n;
        Ok(value)
    }

    fn align(&mut self) {
        self.acc = 0;
        self.count = 0;
    }
}

/// A canonical Huffman code: how many codes of each length, and the
/// symbols in code order.
struct Huffman {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self, InflateError> {
        let mut counts = [0u16; 16];
        for &length in lengths {
            counts[length as usize] += 1;
        }
        counts[0] = 0;
        let mut offsets = [0u16; 16];
        for length in 1..16 {
            offsets[length] = offsets[length - 1] + counts[length - 1];
        }
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &length) in lengths.iter().enumerate() {
            if length != 0 {
                symbols[offsets[length as usize] as usize] = symbol as u16;
                offsets[length as usize] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, bits: &mut Bits<'_>) -> Result<u16, InflateError> {
        let mut code: i32 = 0;
        let mut first: i32 = 0;
        let mut index: i32 = 0;
        for length in 1..16 {
            code |= bits.bits(1)? as i32;
            let count = self.counts[length] as i32;
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(InflateError::Format("code not found".into()))
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Decodes a raw DEFLATE stream.
pub fn inflate(bytes: &[u8]) -> Result<Vec<u8>, InflateError> {
    let mut bits = Bits {
        bytes,
        position: 0,
        acc: 0,
        count: 0,
    };
    let mut out = Vec::new();
    loop {
        let last = bits.bits(1)? == 1;
        match bits.bits(2)? {
            0 => {
                bits.align();
                let header = bytes
                    .get(bits.position..bits.position + 4)
                    .ok_or_else(|| InflateError::Format("stored block header".into()))?;
                let len = u16::from_le_bytes([header[0], header[1]]) as usize;
                let nlen = u16::from_le_bytes([header[2], header[3]]);
                if len as u16 != !nlen {
                    return Err(InflateError::Format("stored block length".into()));
                }
                let start = bits.position + 4;
                let data = bytes
                    .get(start..start + len)
                    .ok_or_else(|| InflateError::Format("stored block ends early".into()))?;
                out.extend_from_slice(data);
                bits.position = start + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                lengths[..144].fill(8);
                lengths[144..256].fill(9);
                lengths[256..280].fill(7);
                lengths[280..].fill(8);
                let literals = Huffman::new(&lengths)?;
                let distances = Huffman::new(&[5u8; 30])?;
                block(&mut bits, &mut out, &literals, &distances)?;
            }
            2 => {
                let hlit = bits.bits(5)? as usize + 257;
                let hdist = bits.bits(5)? as usize + 1;
                let hclen = bits.bits(4)? as usize + 4;
                const ORDER: [usize; 19] = [
                    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                ];
                let mut code_lengths = [0u8; 19];
                for &index in ORDER.iter().take(hclen) {
                    code_lengths[index] = bits.bits(3)? as u8;
                }
                let codes = Huffman::new(&code_lengths)?;
                let mut lengths = vec![0u8; hlit + hdist];
                let mut filled = 0;
                while filled < hlit + hdist {
                    let symbol = codes.decode(&mut bits)?;
                    let (value, repeat) = match symbol {
                        0..=15 => (symbol as u8, 1),
                        16 => {
                            if filled == 0 {
                                return Err(InflateError::Format("repeat with no length".into()));
                            }
                            (lengths[filled - 1], 3 + bits.bits(2)? as usize)
                        }
                        17 => (0, 3 + bits.bits(3)? as usize),
                        _ => (0, 11 + bits.bits(7)? as usize),
                    };
                    if filled + repeat > lengths.len() {
                        return Err(InflateError::Format("too many code lengths".into()));
                    }
                    lengths[filled..filled + repeat].fill(value);
                    filled += repeat;
                }
                let literals = Huffman::new(&lengths[..hlit])?;
                let distances = Huffman::new(&lengths[hlit..])?;
                block(&mut bits, &mut out, &literals, &distances)?;
            }
            _ => return Err(InflateError::Format("block type 3".into())),
        }
        if last {
            return Ok(out);
        }
    }
}

fn block(
    bits: &mut Bits<'_>,
    out: &mut Vec<u8>,
    literals: &Huffman,
    distances: &Huffman,
) -> Result<(), InflateError> {
    loop {
        let symbol = literals.decode(bits)? as usize;
        match symbol {
            0..=255 => out.push(symbol as u8),
            256 => return Ok(()),
            257..=285 => {
                let index = symbol - 257;
                let length =
                    LENGTH_BASE[index] as usize + bits.bits(LENGTH_EXTRA[index] as u32)? as usize;
                let code = distances.decode(bits)? as usize;
                if code >= 30 {
                    return Err(InflateError::Format("distance code".into()));
                }
                let distance =
                    DIST_BASE[code] as usize + bits.bits(DIST_EXTRA[code] as u32)? as usize;
                if distance > out.len() {
                    return Err(InflateError::Format("distance before the start".into()));
                }
                let start = out.len() - distance;
                for k in 0..length {
                    let byte = out[start + k];
                    out.push(byte);
                }
            }
            _ => return Err(InflateError::Format("literal code".into())),
        }
    }
}

/// Adler-32 of `bytes`.
pub fn adler32(bytes: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in bytes.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

/// Decodes a zlib stream and checks its trailer.
pub fn unzlib(bytes: &[u8]) -> Result<Vec<u8>, InflateError> {
    if bytes.len() < 6 {
        return Err(InflateError::Format("too short for zlib".into()));
    }
    if bytes[0] & 0x0f != 8 || !u16::from_be_bytes([bytes[0], bytes[1]]).is_multiple_of(31) {
        return Err(InflateError::Format("zlib header".into()));
    }
    if bytes[1] & 0x20 != 0 {
        return Err(InflateError::Format("preset dictionary".into()));
    }
    let out = inflate(&bytes[2..bytes.len() - 4])?;
    let trailer = u32::from_be_bytes([
        bytes[bytes.len() - 4],
        bytes[bytes.len() - 3],
        bytes[bytes.len() - 2],
        bytes[bytes.len() - 1],
    ]);
    if adler32(&out) != trailer {
        return Err(InflateError::Checksum);
    }
    Ok(out)
}

/// Wraps bytes as a zlib stream of stored blocks: valid everywhere, no
/// smaller than the input.
pub fn zlib_stored(bytes: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let mut chunks = bytes.chunks(65535).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]);
    }
    while let Some(chunk) = chunks.next() {
        out.push(if chunks.peek().is_none() { 1 } else { 0 });
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(bytes).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(dead_code)]
    mod vectors {
        include!("../tests/vectors.rs");
    }
    use vectors::*;

    #[test]
    fn inflates_stored_fixed_and_dynamic_blocks() {
        assert_eq!(unzlib(ZLIB_STORED).unwrap(), b"stored bytes here");
        assert_eq!(unzlib(ZLIB_FIXED).unwrap(), b"abcabcabcabcabcabc");
        let text = unzlib(ZLIB_TEXT).unwrap();
        assert_eq!(text.len(), 2016);
        assert!(text.starts_with(b"the quick brown fox"));
        assert_eq!(
            &text[text.len() - 256..],
            (0..=255).collect::<Vec<u8>>().as_slice()
        );
        let mut broken = ZLIB_TEXT.to_vec();
        let last = broken.len() - 1;
        broken[last] ^= 1;
        assert!(matches!(unzlib(&broken), Err(InflateError::Checksum)));
    }

    #[test]
    fn stored_streams_round_trip() {
        let data: Vec<u8> = (0..70000u32).map(|i| (i * 7) as u8).collect();
        assert_eq!(unzlib(&zlib_stored(&data)).unwrap(), data);
        assert_eq!(unzlib(&zlib_stored(b"")).unwrap(), b"");
    }
}
